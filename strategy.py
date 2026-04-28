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
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily close for price
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-period RSI on daily close for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_6 = 100 - (100 / (1 + rs))
    
    # Align ATR and RSI to 6h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_6)
    
    # Volatility filter: ATR ratio (current vs 20-period average)
    atr_ma = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values
    vol_ratio = atr_aligned / (atr_ma + 1e-10)
    
    # Momentum filter: RSI extremes
    rsi_overbought = rsi_aligned > 70
    rsi_oversold = rsi_aligned < 30
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_ma[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low or high volatility
        vol_filter = (vol_ratio[i] > 0.5) and (vol_ratio[i] < 2.0)
        
        # Volume filter: above average volume
        vol_filter2 = volume[i] > vol_ma[i]
        
        # Entry conditions:
        # Long: RSI oversold + volatility expansion + volume
        # Short: RSI overbought + volatility expansion + volume
        long_entry = rsi_oversold[i] and vol_filter and vol_filter2
        short_entry = rsi_overbought[i] and vol_filter and vol_filter2
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        long_exit = rsi_aligned[i] >= 40
        short_exit = rsi_aligned[i] <= 60
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_RSI6_VolatilityFilter_Session"
timeframe = "6h"
leverage = 1.0