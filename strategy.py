#!/usr/bin/env python3
# Strategy: 1d_1w_WT_LongOnly_Pullback
# Hypothesis: In a strong weekly trend (price above weekly 200 EMA), enter long on daily WT (Williams %R) pullbacks from overbought, with volume confirmation and ATR-based stop.
# Works in bull markets via trend-following pullbacks; avoids bear markets by requiring weekly uptrend filter.
# Target: 15-25 trades/year to minimize fee drag.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend filter (bull market only)
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Load daily data for entries and indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R (14-period) on daily
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    willr = -100 * ((highest_high - close_1d) / rr)
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR for stoploss (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    entry_price = 0.0
    
    for i in range(200, n):  # Start after weekly EMA200 warmup
        # Skip if NaN in critical values
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(willr_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: weekly uptrend (price above weekly EMA200), Williams %R pulls back from oversold (< -80), volume confirmation
            if (price > ema200_1w_aligned[i] and 
                willr_aligned[i] < -80 and  # Oversold condition
                vol > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
        
        elif position == 1:
            # Exit: Williams %R returns to overbought (> -20) or ATR-based stop
            if (willr_aligned[i] > -20 or 
                price < entry_price - 2.0 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "1d_1w_WT_LongOnly_Pullback"
timeframe = "1d"
leverage = 1.0