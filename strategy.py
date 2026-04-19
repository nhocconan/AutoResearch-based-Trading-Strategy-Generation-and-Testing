#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_MultiTimeframe_Momentum_Volume_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for momentum and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d momentum (ROC 3)
    close_1d = df_1d['close'].values
    roc_3 = np.full(len(close_1d), np.nan)
    for i in range(3, len(close_1d)):
        if close_1d[i-3] != 0:
            roc_3[i] = (close_1d[i] - close_1d[i-3]) / close_1d[i-3] * 100
    
    # Calculate 1d EMA34 for trend filter
    close_series = pd.Series(close_1d)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6m volume spike (volume > 1.5 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Align 1d indicators to 6h timeframe
    roc_3_aligned = align_htf_to_ltf(prices, df_1d, roc_3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(roc_3_aligned[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Conditions:
        # Long: positive 1d momentum + price above 1d EMA34 + volume spike
        # Short: negative 1d momentum + price below 1d EMA34 + volume spike
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when bullish momentum + price above EMA + volume spike
            if roc_3_aligned[i] > 0.5 and price_above_ema and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when bearish momentum + price below EMA + volume spike
            elif roc_3_aligned[i] < -0.5 and price_below_ema and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when momentum turns negative or price breaks below EMA
            if roc_3_aligned[i] < 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when momentum turns positive or price breaks above EMA
            if roc_3_aligned[i] > 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals