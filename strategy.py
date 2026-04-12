#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 14-period RSI on weekly data
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:14] = np.nan  # Ensure proper warmup
    
    # Align RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Calculate daily ATR for position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period ATR mean for volatility normalization
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / (atr_ma + 1e-10)
    
    # Volatility-based position sizing (inverse volatility)
    # Higher volatility = smaller position, capped between 0.20 and 0.30
    vol_scaling = np.clip(1.0 / (atr_ratio + 0.001), 0.8, 1.5)
    base_size = 0.25
    position_size = base_size * vol_scaling
    position_size = np.clip(position_size, 0.20, 0.30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if np.isnan(rsi_aligned[i]) or np.isnan(position_size[i]):
            signals[i] = 0.0 if position == 0 else (position_size[i] if position == 1 else -position_size[i])
            continue
        
        # Mean reversion signals based on weekly RSI extremes
        oversold = rsi_aligned[i] < 30
        overbought = rsi_aligned[i] > 70
        
        # Exit when RSI returns to neutral zone
        rsi_neutral = (rsi_aligned[i] >= 40) and (rsi_aligned[i] <= 60)
        
        # Execute trades
        if oversold and position != 1:
            position = 1
            signals[i] = position_size[i]
        elif overbought and position != -1:
            position = -1
            signals[i] = -position_size[i]
        elif rsi_neutral and position != 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position with dynamic sizing
            if position == 1:
                signals[i] = position_size[i]
            elif position == -1:
                signals[i] = -position_size[i]
            else:
                signals[i] = 0.0
    
    return signals