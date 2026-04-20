#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR (14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h Donchian channels (20) for breakout signals
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h RSI (14) for momentum filter
    delta = pd.Series(close_6h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_6h = 100 - (100 / (1 + rs))
    rsi_6h = rsi_6h.fillna(50).values  # fill NaN with neutral 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close_6h[i]
        upper_val = donchian_upper_6h[i]
        lower_val = donchian_lower_6h[i]
        rsi_val = rsi_6h[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(rsi_val) or np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 6h upper Donchian with RSI > 50 and sufficient volatility
            if close_val > upper_val and rsi_val > 50 and atr_1d_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h lower Donchian with RSI < 50 and sufficient volatility
            elif close_val < lower_val and rsi_val < 50 and atr_1d_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 6h lower Donchian or RSI < 40
            if close_val < lower_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 6h upper Donchian or RSI > 60
            if close_val > upper_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_DonchianBreakout_RSIFilter_Volatility
# Enters long when 6h price breaks above 20-period Donchian upper with RSI > 50
# Enters short when 6h price breaks below 20-period Donchian lower with RSI < 50
# Requires daily ATR > 0 for volatility filter
# Exits on opposite Donchian touch or RSI extremes (<40 for long, >60 for short)
# Uses 25% position sizing to manage risk in volatile 6h timeframe
name = "6h_DonchianBreakout_RSIFilter_Volatility"
timeframe = "6h"
leverage = 1.0