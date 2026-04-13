#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based regime filter.
    # Long when price breaks above 20-period Donchian high with volume spike and ATR ratio > 0.8 (trending regime).
    # Short when price breaks below 20-period Donchian low with volume spike and ATR ratio > 0.8.
    # Exit when price crosses 20-period EMA (mean reversion within trend).
    # Uses discrete size 0.25 to minimize fee churn. Target: 100-180 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (same timeframe, but we need rolling window)
    # Since primary is 4h, we calculate Donchian directly on 4h data
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 1d data for volume and ATR regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume mean (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) for regime filter
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    # Calculate 1d ATR ratio (current ATR / 20-period mean ATR) to detect trending vs ranging
    atr_series = pd.Series(atr_1d)
    atr_ma_20 = atr_series.rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / (atr_ma_20 + 1e-10)  # Avoid division by zero
    
    # Align HTF indicators to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_20[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = volume_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: ATR ratio > 0.8 indicates trending market (avoid ranging/choppy)
        trending_regime = atr_ratio_aligned[i] > 0.8
        
        # Entry conditions: price breaks Donchian levels with filters
        long_entry = (close[i] > donchian_high[i] and 
                     volume_confirmation and 
                     trending_regime)
        short_entry = (close[i] < donchian_low[i] and 
                      volume_confirmation and 
                      trending_regime)
        
        # Exit conditions: price crosses 20-period EMA (mean reversion within trend)
        long_exit = close[i] < ema_20[i]
        short_exit = close[i] > ema_20[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_volume_atr_regime_v1"
timeframe = "4h"
leverage = 1.0