#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
# Uses 12h EMA50 for trend direction (long only when price > EMA50, short only when price < EMA50).
# Entry: price breaks above Camarilla R1 level with volume > 2.0x 20-period MA for longs,
#        or breaks below Camarilla S1 level with volume spike for shorts.
# Exit: ATR(14) trailing stop (2.5x ATR) or reversal of 12h EMA50 trend.
# Discrete sizing 0.25. Target: 100-180 total trades over 4 years (25-45/year).
# Camarilla levels from 1d provide robust daily support/resistance; 12h EMA50 filters counter-trend trades;
# volume confirmation reduces false breakouts. Works in bull via trend-following breakouts
# and in bear via short breakdowns with trend alignment.

name = "4h_Camarilla_R1S1_12hEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])  # same length as prices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = close + 1.0833*(high-low), S1 = close - 1.0833*(high-low)
    camarilla_r1_1d = df_1d['close'] + 1.0833 * (df_1d['high'] - df_1d['low'])
    camarilla_s1_1d = df_1d['close'] - 1.0833 * (df_1d['high'] - df_1d['low'])
    # Align to 4h timeframe (wait for 1d bar to close)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d.values)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d.values)
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long ATR stop
    lowest_since_entry = 0.0   # for short ATR stop
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        r1_level = camarilla_r1[i]
        s1_level = camarilla_s1[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Update highest/lowest since entry for ATR stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry != 0 else low[i]
        
        # Entry logic
        if position == 0:
            # Long: break above R1 with volume spike in uptrend
            if close_val > r1_level and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = high[i]
            # Short: break below S1 with volume spike in downtrend
            elif close_val < s1_level and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = low[i]
        elif position == 1:
            # Long exit: ATR stoploss OR price breaks below S1 OR trend turns down
            atr_stop = highest_since_entry - (2.5 * atr_val)
            if close_val < atr_stop or close_val < s1_level or not is_uptrend:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR stoploss OR price breaks above R1 OR trend turns up
            atr_stop = lowest_since_entry + (2.5 * atr_val)
            if close_val > atr_stop or close_val > r1_level or not is_downtrend:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals