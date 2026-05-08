#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WilliamsAlligator_1wTrend_TrendConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d: SMA(13,8), SMA(8,5), SMA(5,3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period SMA shifted 8
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period SMA shifted 5
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period SMA shifted 3
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_conf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price above 1w EMA50 + volume confirmation
            long_cond = (lips[i] > teeth[i]) and (teeth[i] > jaw[i]) and \
                        (close[i] > ema_50_1w_aligned[i]) and volume_conf[i]
            # Short: Jaw > Teeth > Lips (bearish alignment) + price below 1w EMA50 + volume confirmation
            short_cond = (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and \
                         (close[i] < ema_50_1w_aligned[i]) and volume_conf[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment (Jaw > Teeth > Lips)
            if (jaw[i] > teeth[i]) and (teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment (Lips > Teeth > Jaw)
            if (lips[i] > teeth[i]) and (teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals