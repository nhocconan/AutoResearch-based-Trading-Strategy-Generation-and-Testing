#!/usr/bin/env python3
"""
12h Camarilla Pivot + 1d EMA200 Trend + Volume Spike + ATR Stop
Hypothesis: Camarilla pivot levels from daily charts act as strong support/resistance.
In trending markets (price above/below 1d EMA200), price often reacts at these levels.
Volume spike confirms institutional interest. Works in both bull (buying dips at S1/S2) 
and bear (selling rallies at R1/R2) markets. Low-frequency trading to avoid fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate Camarilla levels (using previous day's OHLC)
    # Typical Camarilla: H4 = C + 1.1*(H-L), L4 = C - 1.1*(H-L)
    #                    H3 = C + 1.1*(H-L)/2, L3 = C - 1.1*(H-L)/2
    #                    H2 = C + 1.1*(H-L)/4, L2 = C - 1.1*(H-L)/4
    #                    H1 = C + 1.1*(H-L)/6, L1 = C - 1.1*(H-L)/6
    # We'll use H3, L3, H4, L4 as key levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous close)
    H3 = close_1d + 1.1 * daily_range / 2
    L3 = close_1d - 1.1 * daily_range / 2
    H4 = close_1d + 1.1 * daily_range
    L4 = close_1d - 1.1 * daily_range
    
    # Align to 12h timeframe (these levels are valid for the entire day)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 24-period volume MA on 12h (2 days)
    volume_ma_24 = pd.Series(prices['volume'].values).rolling(window=24, min_periods=24).mean()
    
    # ATR for stop loss and position sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    # Start after warmup period
    start_idx = 200  # enough for EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or np.isnan(H4_12h[i]) or 
            np.isnan(L4_12h[i]) or np.isnan(ema_200_12h[i]) or 
            np.isnan(volume_ma_24.iloc[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = prices['volume'].iloc[i]
        vol_ma = volume_ma_24.iloc[i]
        
        # Volume spike filter: at least 1.5x average volume
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price near support levels in uptrend
            # Uptrend: price above 1d EMA200
            if price > ema_200_12h[i] and volume_ok:
                # Buy near L3 or L4 with some tolerance
                if abs(price - L3_12h[i]) / price < 0.005 or abs(price - L4_12h[i]) / price < 0.005:
                    signals[i] = 0.25
                    position = 1
            
            # Short conditions: price near resistance levels in downtrend
            # Downtrend: price below 1d EMA200
            elif price < ema_200_12h[i] and volume_ok:
                # Sell near H3 or H4 with some tolerance
                if abs(price - H3_12h[i]) / price < 0.005 or abs(price - H4_12h[i]) / price < 0.005:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches resistance or trend changes
            if price >= H3_12h[i] or price < ema_200_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches support or trend changes
            if price <= L3_12h[i] or price > ema_200_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_EMA200_Trend_Volume"
timeframe = "12h"
leverage = 1.0