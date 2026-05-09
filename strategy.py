#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Choppiness_Regime_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(np.sum(tr[-14:]) / (np.log(14) * atr14[-14:])) if len(tr) >= 14 else np.full_like(tr, np.nan)
    # Vectorized version
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (np.log(14) * atr14))
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Trend filter: 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20, 14)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or
            np.isnan(ema50_1d[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        trend = ema50_1d[i]
        price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        
        if position == 0:
            # Mean reversion in ranging market (high chop)
            if chop_val > 61.8:  # Ranging market
                if price <= lower:  # Near lower BB -> long
                    signals[i] = 0.25
                    position = 1
                elif price >= upper:  # Near upper BB -> short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price crosses SMA20 or chop drops (trending)
            if price >= sma20[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses SMA20 or chop drops (trending)
            if price <= sma20[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals