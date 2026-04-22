#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for volatility regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly ATR for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    # 50-period SMA on weekly close for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Load daily data for ATR-based entry trigger (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Daily ATR for volatility breakout
    tr1_d = high_1d[1:] - low_1d[1:]
    tr2_d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Load 12h data for EMA trend filter (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detection (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (14-period) on primary timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema20_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_1w = atr_1w_aligned[i]
        sma50_1w = sma50_1w_aligned[i]
        atr_1d = atr_1d_aligned[i]
        ema20_12h = ema20_12h_aligned[i]
        atr_val = atr[i]
        
        # Regime filter: only trade when weekly ATR is above its 50-period SMA (high volatility regime)
        high_vol_regime = atr_1w > sma50_1w
        
        if position == 0 and high_vol_regime:
            # Long: price breaks above close + 2*ATR(1d) with volume + above 12h EMA20
            if price > close_1d[i] + 2.0 * atr_1d and vol > 1.5 * vol_ma and price > ema20_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below close - 2*ATR(1d) with volume + below 12h EMA20
            elif price < close_1d[i] - 2.0 * atr_1d and vol > 1.5 * vol_ma and price < ema20_12h:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: mean reversion to daily close or ATR stop
            mean_rev_exit = (position == 1 and price < close_1d[i]) or (position == -1 and price > close_1d[i])
            
            # ATR stop loss: 1.5 * ATR from entry
            stop_loss = (position == 1 and price < entry_price - 1.5 * atr_val) or \
                        (position == -1 and price > entry_price + 1.5 * atr_val)
            
            if mean_rev_exit or stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyATRRegime_DailyATRBreakout_12hEMA20_Volume"
timeframe = "12h"
leverage = 1.0