#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate daily ATR for volatility regime
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_daily = pd.Series(atr_daily).rolling(window=30, min_periods=30).mean().values
    volatility_regime = atr_daily / (atr_ma_daily + 1e-10)  # Current ATR vs 30-day average
    
    # Align daily volatility regime to 6h timeframe
    vol_regime_6h = align_htf_to_ltf(prices, daily, volatility_regime)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian channels to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, daily, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, daily, donchian_low)
    donchian_mid_6h = align_htf_to_ltf(prices, daily, donchian_mid)
    
    # Calculate daily volume moving average for confirmation
    volume_ma_daily = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio_daily = daily_volume / (volume_ma_daily + 1e-10)
    
    # Align volume ratio to 6h timeframe
    volume_ratio_6h = align_htf_to_ltf(prices, daily, volume_ratio_daily)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_regime_6h[i]) or np.isnan(donchian_high_6h[i]) or 
            np.isnan(donchian_low_6h[i]) or np.isnan(volume_ratio_6h[i])):
            signals[i] = 0.0
            continue
        
        # Strategy: Donchian breakout with volatility and volume filters
        # Long when price breaks above daily Donchian high in high volatility + high volume
        # Short when price breaks below daily Donchian low in high volatility + high volume
        if (close[i] > donchian_high_6h[i] and 
            vol_regime_6h[i] > 1.2 and  # Above average volatility
            volume_ratio_6h[i] > 1.5):  # Above average volume
            signals[i] = 0.25
        elif (close[i] < donchian_low_6h[i] and 
              vol_regime_6h[i] > 1.2 and  # Above average volatility
              volume_ratio_6h[i] > 1.5):  # Above average volume
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_DailyDonchian_Breakout_Vol_VolRegime"
timeframe = "6h"
leverage = 1.0