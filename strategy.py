#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly 10-period EMA for long-term trend
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate daily RSI(2)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    
    # Calculate daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_10 = pd.Series(volume_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_10_1w_aligned[i]
        rsi_val = rsi_2_aligned[i]
        vol_val = volume_1d[i]
        vol_avg_val = vol_avg_10_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(rsi_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly EMA (uptrend), RSI < 10 (extremely oversold), volume above average
            if close_val > ema_val and rsi_val < 10 and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA (downtrend), RSI > 90 (extremely overbought), volume above average
            elif close_val < ema_val and rsi_val > 90 and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA or RSI > 90 (extremely overbought)
            if close_val < ema_val or rsi_val > 90:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA or RSI < 10 (extremely oversold)
            if close_val > ema_val or rsi_val < 10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_1wEMA10_2RSI_Volume
# Uses 1-week 10-period EMA for long-term trend direction
# Enters long when price above weekly EMA, RSI(2) < 10, and volume above average
# Enters short when price below weekly EMA, RSI(2) > 90, and volume above average
# Exits when price crosses weekly EMA or RSI(2) reaches opposite extreme
# Designed for 1d timeframe with ~10-20 trades/year
name = "1d_1wEMA10_2RSI_Volume"
timeframe = "1d"
leverage = 1.0