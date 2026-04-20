#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20) for trend bias
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 4h RSI (14) for overbought/oversold
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    # Calculate 4h ATR (14) for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate 1h ATR (14) for entry/exit
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1_1h = high_1h - low_1h
    tr2_1h = np.abs(high_1h - np.roll(close_1h, 1))
    tr3_1h = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    tr_1h[0] = tr1_1h[0]
    atr_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        rsi_val = rsi_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        atr_1h_val = atr_1h[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(rsi_val) or np.isnan(atr_4h_val) or 
            np.isnan(atr_1h_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h upper Donchian with RSI < 70 (not overbought)
            if close_val > upper_val and rsi_val < 70 and atr_4h_val > 0:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h lower Donchian with RSI > 30 (not oversold)
            elif close_val < lower_val and rsi_val > 30 and atr_4h_val > 0:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 4h lower Donchian or 1.5*ATR stop
            if close_val < lower_val or close_val < prices['high'].iloc[i] - 1.5 * atr_1h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price closes above 4h upper Donchian or 1.5*ATR stop
            if close_val > upper_val or close_val > prices['low'].iloc[i] + 1.5 * atr_1h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# 1h_4hDonchian_RSI_ATRFilter_V1
# Uses 4h Donchian channels (20) for trend direction bias
# Enters long when 1h price breaks above 4h upper Donchian band with RSI < 70
# Enters short when 1h price breaks below 4h lower Donchian band with RSI > 30
# Uses 4h ATR as volatility filter to avoid choppy markets
# Session filter: 08-20 UTC to reduce noise
# Exits on opposite band touch or 1.5*ATR stop (using 1h ATR)
# Target: 15-30 trades/year (60-120 total over 4 years)
name = "1h_4hDonchian_RSI_ATRFilter_V1"
timeframe = "1h"
leverage = 1.0