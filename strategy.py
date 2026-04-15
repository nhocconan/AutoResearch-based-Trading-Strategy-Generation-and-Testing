#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation
# Works in bull markets by capturing breakouts, in bear markets by filtering with 12h EMA to avoid false signals
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 20-50 trades/year per symbol to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA34
        trend_filter = close[i] > ema_34_12h_aligned[i]
        
        # Long conditions:
        # 1. Price above 12h EMA34 (bullish bias)
        # 2. Price breaks above 4h Donchian(20) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 2.0x average
        if (trend_filter and
            close[i] > donchian_high_20[i] and
            volume_ratio[i] > 2.0):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 12h EMA34 (bearish bias)
        # 2. Price breaks below 4h Donchian(20) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 2.0x average
        elif (not trend_filter and
              close[i] < donchian_low_20[i] and
              volume_ratio[i] > 2.0):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0