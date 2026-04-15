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
    
    # Get daily data for HTF context (once before loop)
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend direction
    daily_ema_50 = pd.Series(daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_50_aligned = align_htf_to_ltf(prices, daily, daily_ema_50)
    
    # Calculate daily ATR for volatility filter
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volatility filter: ATR > 0.25% of price (avoid low volatility chop)
    vol_filter = atr_14d_aligned > (0.0025 * close)
    
    # Calculate daily volume EMA for volume spike detection
    vol_ema_20d = pd.Series(daily['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20d_aligned = align_htf_to_ltf(prices, daily, vol_ema_20d)
    
    # Volume filter: current volume > 1.8x 20-day average volume (strong confirmation)
    vol_threshold = 1.8 * vol_ema_20d_aligned
    vol_spike = volume > vol_threshold
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_ema_50_aligned[i]) or np.isnan(atr_14d_aligned[i]) or 
            np.isnan(vol_ema_20d_aligned[i])):
            continue
        
        # Only trade when volatility is sufficient and volume spike present
        if not vol_filter[i] or not vol_spike[i]:
            signals[i] = 0.0
            continue
            
        # Long: Price above daily EMA50
        if close[i] > daily_ema_50_aligned[i]:
            signals[i] = 0.25
        
        # Short: Price below daily EMA50
        elif close[i] < daily_ema_50_aligned[i]:
            signals[i] = -0.25
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_EMA50_VolumeSpike_VolFilter"
timeframe = "12h"
leverage = 1.0