#!/usr/bin/env python3
"""
4h_Alligator_Momentum_Breakout_v1
Hypothesis: Williams Alligator identifies trends (price > red line = uptrend, price < red line = downtrend). 
Breakouts above/below the Alligator mouth (green/red lines) with volume confirmation signal trend continuation.
Works in bull by catching breakouts in uptrends; works in bear by catching breakdowns in downtrends.
Uses 1d ADX filter to avoid choppy markets. Targets 20-30 trades/year with tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future shifts"""
    # Smoothed Moving Average (approximated with EMA for simplicity)
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().values
    return jaw, teeth, lips

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
    
    dx = np.zeros_like(close)
    divisor = plus_di + minus_di
    mask = divisor != 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / divisor[mask]
    
    adx = np.zeros_like(close)
    if len(dx) >= period:
        adx[period-1] = np.mean(dx[:period])
    
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for Alligator
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Alligator
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    jaw, teeth, lips = calculate_alligator(high_4h, low_4h, close_4h)
    
    # Align Alligator components (Jaw is the main trend indicator)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Load 1d data once for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for indicators to stabilize
        # Skip if indicators not ready
        if np.isnan(jaw_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: ADX > 20 indicates trending market
        trend_filter = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long: price above Alligator teeth (bullish alignment) + volume + trend
            if (price > teeth_aligned[i] and price > lips_aligned[i] and 
                volume_ok and trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below Alligator teeth (bearish alignment) + volume + trend
            elif (price < teeth_aligned[i] and price < lips_aligned[i] and 
                  volume_ok and trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below Alligator lips (trend weakness) or ADX < 15 (losing trend)
            if price < lips_aligned[i] or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above Alligator lips (trend weakness) or ADX < 15 (losing trend)
            if price > lips_aligned[i] or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Alligator_Momentum_Breakout_v1"
timeframe = "4h"
leverage = 1.0