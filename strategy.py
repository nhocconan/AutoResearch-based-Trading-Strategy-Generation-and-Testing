#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h ATR-based breakout with volume confirmation and session filter.
- Uses 4h ATR(14) to measure volatility and set breakout levels from the current open
- Enter long when price breaks above open + 0.5 * 4h ATR with volume > 1.5x 20-period volume MA
- Enter short when price breaks below open - 0.5 * 4h ATR with volume > 1.5x 20-period volume MA
- Exit when price returns to the open level (mean reversion to session open)
- Only trade during 08:00-20:00 UTC to avoid low-liquidity hours
- Fixed position size 0.20 to manage drawdown
- Uses 4h trend filter to avoid counter-trend trades
- Designed for 1h timeframe with strict entry conditions to limit trades to 60-150 total over 4 years
"""

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
    
    # Get 4-hour data for ATR and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h ATR(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        atr_val = atr_14_aligned[i]
        ema_val = ema_50_aligned[i]
        open_price = prices['open'].iloc[i]
        
        # Calculate breakout levels
        upper_breakout = open_price + 0.5 * atr_val
        lower_breakout = open_price - 0.5 * atr_val
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price breaks above upper_breakout + volume spike + price above 4h EMA50
            if price > upper_breakout and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower_breakout + volume spike + price below 4h EMA50
            elif price < lower_breakout and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when price returns to open level (mean reversion)
            if price <= open_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when price returns to open level (mean reversion)
            if price >= open_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_ATRBreakout_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0