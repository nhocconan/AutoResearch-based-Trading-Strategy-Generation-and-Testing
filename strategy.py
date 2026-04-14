#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate daily ATR for volatility filtering
    atr_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-period average volume for confirmation
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=12, min_periods=12).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 12)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price pulls back to EMA200 in uptrend with volume confirmation
            if (price > ema200_aligned[i] and 
                abs(price - ema200_aligned[i]) < 0.5 * atr_aligned[i] and 
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price rallies to EMA200 in downtrend with volume confirmation
            elif (price < ema200_aligned[i] and 
                  abs(price - ema200_aligned[i]) < 0.5 * atr_aligned[i] and 
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price moves away from EMA200 or opposite signal
            if price < ema200_aligned[i] - atr_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price moves away from EMA200 or opposite signal
            if price > ema200_aligned[i] + atr_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_EMA200_Pullback_Volume"
timeframe = "12h"
leverage = 1.0