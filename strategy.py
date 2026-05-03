#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction and 1d RSI extremes for mean reversion entries
# In bull markets: 4h uptrend + 1d RSI < 30 → long (buy dip in uptrend)
# In bear markets: 4h downtrend + 1d RSI > 70 → short (sell rally in downtrend)
# Uses session filter (08-20 UTC) to reduce noise. Target: 20-50 trades/year to minimize fee drag.
# Supertrend identifies trend with built-in ATR-based stop, reducing whipsaws.
# RSI extremes provide high-probability mean reversion entries in the direction of higher timeframe trend.

name = "1h_Supertrend4h_RSI1d_MeanReversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC) to avoid datetime64 issues in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Supertrend (trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = pd.Series(df_4h['high']).diff().abs()
    tr2 = (pd.Series(df_4h['high']) - pd.Series(df_4h['close'].shift())).abs()
    tr3 = (pd.Series(df_4h['low']) - pd.Series(df_4h['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.full(len(df_4h), np.nan, dtype=float)
    direction = np.full(len(df_4h), 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_4h)):
        if i < atr_period:
            continue
            
        # Upper band logic
        if df_4h['close'].iloc[i-1] <= upperband[i-1]:
            upperband[i] = min(upperband[i], upperband[i-1])
        else:
            upperband[i] = hl2[i] + (multiplier * atr[i])
            
        # Lower band logic
        if df_4h['close'].iloc[i-1] >= lowerband[i-1]:
            lowerband[i] = max(lowerband[i], lowerband[i-1])
        else:
            lowerband[i] = hl2[i] - (multiplier * atr[i])
            
        # Supertrend and direction
        if supertrend[i-1] == upperband[i-1]:
            if df_4h['close'].iloc[i] <= upperband[i]:
                supertrend[i] = upperband[i]
                direction[i] = -1
            else:
                supertrend[i] = lowerband[i]
                direction[i] = 1
        else:
            if df_4h['close'].iloc[i] >= lowerband[i]:
                supertrend[i] = lowerband[i]
                direction[i] = 1
            else:
                supertrend[i] = upperband[i]
                direction[i] = -1
    
    # Align Supertrend direction to 1h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, direction.astype(float))
    
    # Get 1d data for RSI (mean reversion signal)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d RSI (14-period)
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align 1d RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for indicators
        # Skip if outside session or any value is NaN
        if (not in_session[i] or 
            np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: 4h uptrend (Supertrend direction = 1) + 1d RSI oversold (< 30)
            if supertrend_direction_aligned[i] == 1 and rsi_1d_aligned[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (Supertrend direction = -1) + 1d RSI overbought (> 70)
            elif supertrend_direction_aligned[i] == -1 and rsi_1d_aligned[i] > 70:
                signals[i] = -0.20
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: 4h trend turns down OR 1d RSI becomes overbought (> 70)
            if supertrend_direction_aligned[i] == -1 or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend turns up OR 1d RSI becomes oversold (< 30)
            if supertrend_direction_aligned[i] == 1 or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals