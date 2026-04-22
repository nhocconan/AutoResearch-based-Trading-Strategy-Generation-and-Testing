#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Supertrend (ATR=10, multiplier=3)
    # ATR calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic upper and lower bands
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + 3 * atr
    lower_band = hl2 - 3 * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close_4h, np.nan)
    direction = np.full_like(close_4h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close_4h)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            continue
            
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else lower_band[i]
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1]) if not np.isnan(supertrend[i-1]) else upper_band[i]
    
    # Align Supertrend to 1h
    supertrend_1h = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_1h = align_htf_to_ltf(prices, df_4h, direction)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI for entry timing
    close_series = prices['close']
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(supertrend_1h[i]) or 
            np.isnan(direction_1h[i]) or 
            np.isnan(ema50_1h[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        st = supertrend_1h[i]
        dir_4h = direction_1h[i]
        ema50 = ema50_1h[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: 4h uptrend + price above Supertrend + above EMA50 + RSI < 70
            if dir_4h == 1 and price > st and price > ema50 and rsi_val < 70:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + price below Supertrend + below EMA50 + RSI > 30
            elif dir_4h == -1 and price < st and price < ema50 and rsi_val > 30:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: trend change or RSI extreme
            exit_signal = False
            
            if position == 1:  # long
                if dir_4h == -1 or price < st or rsi_val > 80:
                    exit_signal = True
            elif position == -1:  # short
                if dir_4h == 1 or price > st or rsi_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Supertrend_EMA50_RSI_Session"
timeframe = "1h"
leverage = 1.0