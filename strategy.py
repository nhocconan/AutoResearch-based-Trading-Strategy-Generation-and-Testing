#!/usr/bin/env python3
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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ATR(10) for volatility regime
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly RSI(14) for momentum filter
    delta = pd.Series(df_1w['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Donchian breakout conditions
    upper_break = np.zeros(n, dtype=bool)
    lower_break = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        upper_break[i] = close[i] > highest_high[i-1]
        lower_break[i] = close[i] < lowest_low[i-1]
    
    signals = np.zeros(n)
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_10_1w_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when weekly ATR is elevated (> 0.8% of price)
        vol_filter = atr_10_1w_aligned[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. 6h Donchian upper breakout
        # 2. Price above weekly EMA20 (bullish bias)
        # 3. Weekly RSI between 30 and 70 (avoid extremes)
        # 4. Volatility filter
        if (upper_break[i] and
            close[i] > ema_20_1w_aligned[i] and
            30 <= rsi_14_1w_aligned[i] <= 70 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h Donchian lower breakout
        # 2. Price below weekly EMA20 (bearish bias)
        # 3. Weekly RSI between 30 and 70 (avoid extremes)
        # 4. Volatility filter
        elif (lower_break[i] and
              close[i] < ema_20_1w_aligned[i] and
              30 <= rsi_14_1w_aligned[i] <= 70 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchian20_EMA20_RSI14_VolFilter_v1"
timeframe = "6h"
leverage = 1.0