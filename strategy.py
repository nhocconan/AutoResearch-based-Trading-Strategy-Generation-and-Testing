#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume confirmation (>1.8x avg volume). Uses ATR(14) trailing stop (2.0x) for risk control. Discrete sizing 0.28.
# Target: 80-180 total trades over 4 years (20-45/year) on 4h timeframe.
# Weekly EMA34 provides strong trend filter to avoid counter-trend trades in ranging markets. Camarilla R4/S4 levels represent significant support/resistance from prior weekly range.
# Volume confirmation ensures institutional participation. Works in bull markets via trend-following breakouts and in bear markets via shorting breakdowns with trend filter.

name = "4h_Camarilla_R4_S4_Breakout_1wEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Camarilla pivot levels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 4h timeframe (wait for 1w bar to close)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla pivot levels from prior 1w bar
    # Camarilla: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_upper = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_lower = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for 1w bar to close)
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1w, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1w, camarilla_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R4 AND 1w EMA34 > 0 (rising trend) AND volume > 1.8x average
            if (close[i] > camarilla_upper_aligned[i] and 
                ema34_1w_aligned[i] > np.roll(ema34_1w_aligned, 1)[i] and  # EMA34 rising
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.28
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Camarilla S4 AND 1w EMA34 < 0 (falling trend) AND volume > 1.8x average
            elif (close[i] < camarilla_lower_aligned[i] and 
                  ema34_1w_aligned[i] < np.roll(ema34_1w_aligned, 1)[i] and  # EMA34 falling
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.28
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.28
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.28
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals