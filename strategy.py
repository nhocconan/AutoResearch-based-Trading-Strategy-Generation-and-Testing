#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d multi-timeframe confluence with volume confirmation
# Uses 4h trend (EMA50), 1d Donchian breakout, and 1h volume spike for entry timing
# Designed to work in both bull/bear by requiring strong volume confirmation on breakouts
# Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag
# Signal size: 0.20 (discrete levels to reduce churn)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d Donchian(20) channels for breakout levels
    donchian_high_20_1d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20_1d)
    donchian_low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20_1d)
    
    # 1h volume ratio (current vs 20-period average) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC (precompute hours array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donchian_high_20_1d_aligned[i]) or 
            np.isnan(donchian_low_20_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above 4h EMA50 (bullish bias)
        # 2. Price breaks above 1d Donchian(20) high
        # 3. Volume confirmation: volume > 2.0x average
        if (close[i] > ema_50_4h_aligned[i] and
            close[i] > donchian_high_20_1d_aligned[i] and
            volume_ratio[i] > 2.0):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. Price below 4h EMA50 (bearish bias)
        # 2. Price breaks below 1d Donchian(20) low
        # 3. Volume confirmation: volume > 2.0x average
        elif (close[i] < ema_50_4h_aligned[i] and
              close[i] < donchian_low_20_1d_aligned[i] and
              volume_ratio[i] > 2.0):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4hEMA50_1dDonchian20_VolumeBreakout_v1"
timeframe = "1h"
leverage = 1.0