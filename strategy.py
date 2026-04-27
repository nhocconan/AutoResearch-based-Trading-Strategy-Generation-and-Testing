#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Get 1d data for Bollinger Bands (20, 2)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = sma_20_1d + 2 * std_20_1d
    bb_lower_1d = sma_20_1d - 2 * std_20_1d
    bb_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_1d)
    bb_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_1d)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(200, vol_period, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(bb_upper_1d_aligned[i]) or
            np.isnan(bb_lower_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 4h EMA200
        uptrend = price > ema_200_4h_aligned[i]
        downtrend = price < ema_200_4h_aligned[i]
        
        # Volatility filter: current ATR < 1.5 * daily ATR (low volatility regime)
        low_volatility = atr_14_1d_aligned[i] > 0 and (price * 0.01) < (1.5 * atr_14_1d_aligned[i])
        
        # Bollinger Band squeeze: bandwidth < 5% of price (indicating low volatility compression)
        bb_width = bb_upper_1d_aligned[i] - bb_lower_1d_aligned[i]
        bb_squeeze = bb_width > 0 and (bb_width / price) < 0.05
        
        if position == 0:
            # Long entry: price breaks above upper BB in uptrend with volume and low volatility squeeze
            if uptrend and price > bb_upper_1d_aligned[i] and vol_ratio > 2.0 and bb_squeeze and low_volatility:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower BB in downtrend with volume and low volatility squeeze
            elif downtrend and price < bb_lower_1d_aligned[i] and vol_ratio > 2.0 and bb_squeeze and low_volatility:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower BB or trend reverses
            if price < bb_lower_1d_aligned[i] or price < ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper BB or trend reverses
            if price > bb_upper_1d_aligned[i] or price > ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Squeeze_Breakout_EMA200_Volume"
timeframe = "4h"
leverage = 1.0