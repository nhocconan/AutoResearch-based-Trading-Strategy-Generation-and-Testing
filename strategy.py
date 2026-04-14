# Solution
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
    
    # Load 1d and 12h data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 1d Williams %R(14) for momentum
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min().values
    willr_1d = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    willr_1d_aligned = align_htf_to_ltf(prices, df_1d, willr_1d)
    
    # Calculate 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(willr_1d_aligned[i]) or 
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price > 1d EMA20 for long, price < 1d EMA20 for short
        trend_filter_long = price > ema_20_1d_aligned[i]
        trend_filter_short = price < ema_20_1d_aligned[i]
        
        # Momentum filter: Williams %R between -80 and -20 (not extreme)
        mom_filter = (willr_1d_aligned[i] >= -80) & (willr_1d_aligned[i] <= -20)
        
        # Volatility filter: ATR > 0.5 * price (avoid low volatility chop)
        vol_filter = atr_12h_aligned[i] > 0.5 * price
        
        if position == 0:
            # Long setup: price above 1d EMA20 + momentum filter + volatility filter
            if trend_filter_long and mom_filter and vol_filter:
                position = 1
                signals[i] = position_size
            # Short setup: price below 1d EMA20 + momentum filter + volatility filter
            elif trend_filter_short and mom_filter and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d EMA20 OR Williams %R > -10 (overbought)
            if price < ema_20_1d_aligned[i] or willr_1d_aligned[i] > -10:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d EMA20 OR Williams %R < -90 (oversold)
            if price > ema_20_1d_aligned[i] or willr_1d_aligned[i] < -90:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12hATR_1dWilliamsR_EMA20_v1"
timeframe = "6h"
leverage = 1.0