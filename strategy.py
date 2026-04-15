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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Williams %R(14) for momentum extremes
    highest_high = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r, additional_delay_bars=0)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h ATR(14) for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 6h ATR > 0.5% of price
        vol_filter = atr_14[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. Price above 12h EMA50 (bullish bias)
        # 2. Williams %R oversold (< -80) - extreme bearish momentum ready to reverse
        # 3. Volatility filter
        if (close[i] > ema_50_12h_aligned[i] and
            williams_r_aligned[i] < -80 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 12h EMA50 (bearish bias)
        # 2. Williams %R overbought (> -20) - extreme bullish momentum ready to reverse
        # 3. Volatility filter
        elif (close[i] < ema_50_12h_aligned[i] and
              williams_r_aligned[i] > -20 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0