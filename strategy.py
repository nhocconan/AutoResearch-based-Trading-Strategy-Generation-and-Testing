#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volatility filter
# Long when price < Bollinger lower (20,2.0) AND 4h EMA(50) rising AND 1d ATR ratio > 1.5
# Short when price > Bollinger upper (20,2.0) AND 4h EMA(50) falling AND 1d ATR ratio > 1.5
# Exit when price crosses Bollinger middle (20-period SMA)
# Uses Bollinger Bands for mean reversion, 4h EMA for trend direction, 1d ATR for volatility regime
# Target: 60-150 total trades over 4 years (15-37/year) for optimal 1h performance

name = "1h_bb20_4h_ema50_1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2.0)
    close_s = pd.Series(close)
    sma20 = close_s.rolling(window=20, min_periods=20).mean()
    std20 = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = (sma20 + 2.0 * std20).values
    bb_lower = (sma20 - 2.0 * std20).values
    bb_middle = sma20.values
    
    # 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    ema4h_50 = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema4h_50)
    
    # 1d ATR(14) for volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value is NaN
    
    # ATR(14)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # 1d ATR ratio: current ATR / 50-period average ATR
    atr_ma50 = pd.Series(atr14_aligned).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14_aligned / atr_ma50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or
            np.isnan(ema4h_50_aligned[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Bollinger middle
        if position == 1:  # long position
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volatility regime
            # Long: price < BB lower AND 4h EMA rising AND ATR ratio > 1.5 (high vol regime)
            if (close[i] < bb_lower[i] and 
                ema4h_50_aligned[i] > ema4h_50_aligned[i-1] and 
                atr_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: price > BB upper AND 4h EMA falling AND ATR ratio > 1.5 (high vol regime)
            elif (close[i] > bb_upper[i] and 
                  ema4h_50_aligned[i] < ema4h_50_aligned[i-1] and 
                  atr_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
    
    return signals