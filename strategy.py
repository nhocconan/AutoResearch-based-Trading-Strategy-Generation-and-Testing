#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout direction with volume confirmation
# and 1d ATR-based volatility filter. Uses 4h trend direction (Donchian breakout)
# for signal bias, volume spike for confirmation, and 1d ATR filter to avoid
# high volatility periods. Designed for low trade frequency (15-37/year) by
# requiring multiple confirmations. Works in bull markets via trend following
# and in bear markets via volatility-filtered mean reversion at extremes.

name = "1h_donchian_breakout_vol_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian breakout trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian upper and lower
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d ATR to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)  # 50% above average
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid high volatility periods
        vol_ratio = atr_1d_aligned[i] / atr_ma_1d_aligned[i] if atr_ma_1d_aligned[i] > 0 else 1.0
        if vol_ratio > 2.0:  # Skip if volatility more than 2x average
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h Donchian breakout
        # Long if price breaks above 20-period high
        # Short if price breaks below 20-period low
        bullish_breakout = close[i] > donch_high_4h_aligned[i]
        bearish_breakout = close[i] < donch_low_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Generate signals with volume confirmation
        if bullish_breakout and vol_confirm:
            signals[i] = 0.25  # 25% long
        elif bearish_breakout and vol_confirm:
            signals[i] = -0.25  # 25% short
        else:
            signals[i] = 0.0
    
    return signals