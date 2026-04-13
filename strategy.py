#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + volume confirmation with 1d trend filter.
    # Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    # Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) with volume spike and 1d uptrend.
    # Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) with volume spike and 1d downtrend.
    # Exit when momentum deteriorates (power crosses zero) or opposing signal.
    # Uses 1d EMA(50) for trend filter. Discrete size 0.25 to minimize fee churn.
    # Target: 50-150 total trades over 4 years (12-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h EMA(13) for Elder Ray
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = ema_13 - low   # Bear Power = EMA(13) - Low
    
    # Calculate 6h volume mean (20-period) with min_periods
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = volume[i] > 1.5 * vol_ma_20[i]
        
        # 1d trend filter
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Entry conditions: Elder Ray momentum with volume confirmation and trend filter
        long_entry = (bull_power[i] > 0 and bear_power[i] < 0 and volume_confirmation and uptrend)
        short_entry = (bear_power[i] > 0 and bull_power[i] < 0 and volume_confirmation and downtrend)
        
        # Exit conditions: momentum deteriorates (power crosses zero) or opposing signal
        long_exit = (bull_power[i] <= 0 or bear_power[i] >= 0)
        short_exit = (bear_power[i] <= 0 or bull_power[i] >= 0)
        
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

name = "6h_1d_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0