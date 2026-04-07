#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily volume confirmation and volatility filter
# Uses weekly Donchian(20) breakout for trend direction, daily volume surge for confirmation,
# and ATR-based volatility filter to avoid choppy markets. Designed for very low trade frequency
# (target: 5-15 trades/year) to minimize fee drag. Works in bull markets via breakout continuation
# and in bear markets via mean reversion at channel extremes.

name = "weekly_donchian20_daily_volume_vol_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Rolling max/min for Donchian channels
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (shifted by 1 for completed weekly bars only)
    donchian_high = align_htf_to_ltf(prices, df_1w, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_min)
    
    # Daily volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid trading in extremely high volatility
        vol_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        vol_filter = vol_ratio < 2.0  # Only trade when volatility is below 2x average
        
        # Volume confirmation: require volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Long breakout: price breaks above weekly Donchian high
        long_breakout = close[i] > donchian_high[i] and vol_filter and vol_confirm
        
        # Short breakout: price breaks below weekly Donchian low
        short_breakout = close[i] < donchian_low[i] and vol_filter and vol_confirm
        
        if long_breakout:
            signals[i] = 0.25
        elif short_breakout:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals