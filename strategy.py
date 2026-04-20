#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 14-day ATR for volatility filter and stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4-hour ATR for stoploss (more responsive)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_aligned[i]
        atr_14_val = atr_14_aligned[i]
        atr_4h_val = atr_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(atr_14_val) or 
            np.isnan(atr_4h_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 50-day EMA with sufficient volatility
            if close_val > ema_val and atr_14_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below 50-day EMA with sufficient volatility
            elif close_val < ema_val and atr_14_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 50-day EMA or ATR-based stop
            if close_val < ema_val or close_val < prices['high'].iloc[i] - 2.0 * atr_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 50-day EMA or ATR-based stop
            if close_val > ema_val or close_val > prices['low'].iloc[i] + 2.0 * atr_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_50EMA_ATRFilter_V1
# Uses 1-day 50-period EMA for trend filter
# Enters long when 4h price is above 1d EMA50
# Enters short when 4h price is below 1d EMA50
# Uses 1-day ATR as volatility filter to avoid choppy markets
# Exits on EMA cross or 2*ATR stoploss (using 4h ATR)
# Designed for 4h timeframe with ~25-50 trades/year
name = "4h_50EMA_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0