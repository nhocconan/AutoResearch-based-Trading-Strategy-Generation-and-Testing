#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 1w EMA34 uptrend and volume > 2.0x average.
# Short when price breaks below S3 with 1w EMA34 downtrend and volume > 2.0x average.
# Uses ATR(14) trailing stop (2.5x) for risk control. Discrete sizing 0.25.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.
# Works in bull via breakout momentum, in bear via mean reversion at extreme levels + volume confirmation.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_ATRStop_v1"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla levels and EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels: R3/S3 = C ± (H-L)*1.1/4
    camarilla_range = (high_1w - low_1w) * 1.1 / 4
    r3 = close_1w + camarilla_range
    s3 = close_1w - camarilla_range
    
    # Align Camarilla levels to 1d timeframe (wait for weekly bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe (wait for weekly bar to close)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND 1w EMA34 uptrend AND volume > 2.0x average
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and  # Price above 1w EMA34 confirms uptrend
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below S3 AND 1w EMA34 downtrend AND volume > 2.0x average
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and  # Price below 1w EMA34 confirms downtrend
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
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
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals