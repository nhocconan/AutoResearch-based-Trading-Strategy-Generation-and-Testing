#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_MultiTF_Trend_Squeeze_With_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_up = ema_34_1d > np.roll(ema_34_1d, 1)
    ema_34_1d_down = ema_34_1d < np.roll(ema_34_1d, 1)
    
    # 1d ATR for volatility normalization
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_ma50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_14 / atr_14_ma50
    
    # 6s Bollinger Bands for squeeze detection
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_bb + bb_std * std_bb
    bb_lower = sma_bb - bb_std * std_bb
    bb_width = (bb_upper - bb_lower) / sma_bb
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_squeeze = bb_width < 0.5 * bb_width_ma  # Bollinger Band squeeze
    
    # Align 1d indicators to 6h timeframe
    ema_34_1d_up_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_up)
    ema_34_1d_down_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_down)
    volatility_ratio_aligned = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_up_aligned[i]) or np.isnan(ema_34_1d_down_aligned[i]) or 
            np.isnan(volatility_ratio_aligned[i]) or np.isnan(bb_squeeze_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend + low volatility + volatility expansion signal
            long_cond = (ema_34_1d_up_aligned[i] and 
                        volatility_ratio_aligned[i] < 1.2 and  # Low volatility
                        bb_squeeze_aligned[i])  # Bollinger squeeze
            
            # Short: 1d downtrend + low volatility + volatility expansion signal
            short_cond = (ema_34_1d_down_aligned[i] and 
                         volatility_ratio_aligned[i] < 1.2 and  # Low volatility
                         bb_squeeze_aligned[i])  # Bollinger squeeze
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d trend reversal or volatility expansion
            if (not ema_34_1d_up_aligned[i] or 
                volatility_ratio_aligned[i] > 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d trend reversal or volatility expansion
            if (not ema_34_1d_down_aligned[i] or 
                volatility_ratio_aligned[i] > 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals