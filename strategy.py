#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-week EMA13 trend filter and volume confirmation
# Elder Ray Power = Close - EMA13 (Bull Power) or EMA13 - Close (Bear Power)
# Long when Bull Power > 0, weekly EMA13 slope > 0 (uptrend), and volume > 1.5x 6s average volume
# Short when Bear Power > 0, weekly EMA13 slope < 0 (downtrend), and volume > 1.5x 6s average volume
# Exit when power reverses sign or opposite signal occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_elder_ray_1w_ema13_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6s data for EMA13 calculation (Elder Ray)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False).mean().values
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    # Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    bull_power = close - ema_13_6h_aligned
    bear_power = ema_13_6h_aligned - close
    
    # 1w data for EMA13 trend filter (slope)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate slope of weekly EMA13 (change over 1 period)
    ema_13_1w_slope = np.diff(ema_13_1w_aligned, prepend=ema_13_1w_aligned[0])
    
    # 6s volume average for confirmation
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_13_1w_slope[i]) or np.isnan(volume_ma_6h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power <= 0 or weekly EMA13 slope <= 0 (trend weakness)
            elif bull_power[i] <= 0 or ema_13_1w_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power <= 0 or weekly EMA13 slope >= 0 (trend weakness)
            elif bear_power[i] <= 0 or ema_13_1w_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: Bull Power > 0, weekly EMA13 slope > 0 (uptrend), volume spike
            if (bull_power[i] > 0 and
                ema_13_1w_slope[i] > 0 and
                volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power > 0, weekly EMA13 slope < 0 (downtrend), volume spike
            elif (bear_power[i] > 0 and
                  ema_13_1w_slope[i] < 0 and
                  volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals