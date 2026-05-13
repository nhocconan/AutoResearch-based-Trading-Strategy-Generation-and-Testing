#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Long when Bull Power > 0, 1d close > 1d EMA34 (uptrend), and volume > 1.5x 20-period average.
# Short when Bear Power < 0, 1d close < 1d EMA34 (downtrend), and volume > 1.5x 20-period average.
# Uses ATR(14) trailing stop (2.0x) for risk control.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.

name = "6h_ElderRay_BullBearPower_1dTrend_Volume_v1"
timeframe = "6h"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA34, Bear Power = Low - EMA34
    bull_power = high_1d - ema_34_1d
    bear_power = low_1d - ema_34_1d
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND 1d close > 1d EMA34 (uptrend) AND volume confirmation
            if bull_power_aligned[i] > 0 and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Bear Power < 0 AND 1d close < 1d EMA34 (downtrend) AND volume confirmation
            elif bear_power_aligned[i] < 0 and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
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
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
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
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
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