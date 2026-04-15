#!/usr/bin/env python3
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
    
    # Get daily data for HTF context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate Donchian channels (20-day) on daily timeframe
    donchian_upper = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_4h = align_htf_to_ltf(prices, daily, donchian_upper)
    donchian_lower_4h = align_htf_to_ltf(prices, daily, donchian_lower)
    
    # Calculate 4h ATR for volatility filter
    close_prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - close_prev),
                               np.abs(low - close_prev)))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio_4h = atr_4h / close
    
    # Calculate daily volume ratio (current vs 20-day average)
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = daily_volume / (vol_ma_20 + 1e-10)
    volume_ratio_4h = align_htf_to_ltf(prices, daily, volume_ratio)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(atr_ratio_4h[i]) or np.isnan(volume_ratio_4h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout strategy with volume confirmation and volatility filter
        # Long when price breaks above daily Donchian upper with volume spike
        if (close[i] > donchian_upper_4h[i] and 
            volume_ratio_4h[i] > 1.5 and  # Volume confirmation
            atr_ratio_4h[i] > 0.01):      # Minimum volatility filter
            signals[i] = 0.25
        # Short when price breaks below daily Donchian lower with volume spike
        elif (close[i] < donchian_lower_4h[i] and 
              volume_ratio_4h[i] > 1.5 and  # Volume confirmation
              atr_ratio_4h[i] > 0.01):      # Minimum volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchian_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0