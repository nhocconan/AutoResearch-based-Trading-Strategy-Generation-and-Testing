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
    
    # Get weekly data for trend and trend strength (HTF)
    weekly = get_htf_data(prices, '1w')
    if len(weekly) < 50:
        return np.zeros(n)
    
    weekly_close = weekly['close'].values
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    
    # Weekly EMA200 for trend direction
    ema200_weekly = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Weekly ATR for trend strength
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_weekly = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly indicators to daily
    ema200_aligned = align_htf_to_ltf(prices, weekly, ema200_weekly)
    atr_aligned = align_htf_to_ltf(prices, weekly, atr_weekly)
    
    # Daily Donchian breakout with volume confirmation
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long: price breaks above Donchian high in uptrend (close > weekly EMA200)
            if close[i] > donchian_high[i] and close[i] > ema200_aligned[i]:
                signals[i] = 0.25
            # Short: price breaks below Donchian low in downtrend (close < weekly EMA200)
            elif close[i] < donchian_low[i] and close[i] < ema200_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Donchian_WeeklyEMA200_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0