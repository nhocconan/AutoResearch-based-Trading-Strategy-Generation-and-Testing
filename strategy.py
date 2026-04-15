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
    vol_regime = atr_daily / (atr_ma_daily + 1e-10)  # >1 = high volatility
    
    # Align volatility regime to 6h
    vol_regime_6h = align_htf_to_ltf(prices, daily, vol_regime)
    
    # Calculate 12h data for trend direction
    htf_12h = get_htf_data(prices, '12h')
    htf_close_12h = htf_12h['close'].values
    
    # Calculate 12h EMA20 for trend
    ema_12h = pd.Series(htf_close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, htf_12h, ema_12h)
    
    # Calculate 12h Donchian(20) for breakout levels
    donch_high_12h = pd.Series(htf_close_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(htf_close_12h).rolling(window=20, min_periods=20).min().values
    donch_high_12h_aligned = align_htf_to_ltf(prices, htf_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, htf_12h, donch_low_12h)
    
    # Calculate daily volume average for confirmation
    vol_ma_daily = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = daily_volume / (vol_ma_daily + 1e-10)
    vol_ratio_6h = align_htf_to_ltf(prices, daily, vol_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_regime_6h[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in high volatility regimes (vol_ratio > 1.2)
        if vol_regime_6h[i] <= 1.2:
            signals[i] = 0.0
            continue
        
        # Long: price breaks above 12h Donchian high + above EMA + volume confirmation
        if (close[i] > donch_high_12h_aligned[i] and 
            close[i] > ema_12h_aligned[i] and
            vol_ratio_6h[i] > 1.5):
            signals[i] = 0.25
        
        # Short: price breaks below 12h Donchian low + below EMA + volume confirmation
        elif (close[i] < donch_low_12h_aligned[i] and 
              close[i] < ema_12h_aligned[i] and
              vol_ratio_6h[i] > 1.5):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_VolRegime_DonchianBreakout_EMAFilter"
timeframe = "6h"
leverage = 1.0