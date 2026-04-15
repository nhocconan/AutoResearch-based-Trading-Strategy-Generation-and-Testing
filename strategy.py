#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation and ADX Filter
# Long when price breaks above weekly Donchian high (20-period) with volume > 1.5x 20-day average and ADX > 25
# Short when price breaks below weekly Donchian low with same conditions
# Uses discrete position sizing (0.25) to limit overtrading and manage drawdown
# Weekly trend filter ensures alignment with higher-timeframe momentum

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (already accounts for weekly bar completion)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly ADX for trend strength
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    tr = np.maximum(high_1w[1:] - low_1w[1:], np.absolute(high_1w[1:] - close_1w[:-1]), np.absolute(low_1w[1:] - close_1w[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above weekly Donchian high, volume spike, ADX > 25
        if (close[i] > donchian_high_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: price breaks below weekly Donchian low, volume spike, ADX > 25
        elif (close[i] < donchian_low_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              adx_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price returns inside Donchian channel or ADX weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= donchian_high_aligned[i] or adx_aligned[i] <= 25)) or
               (signals[i-1] == -0.25 and (close[i] >= donchian_low_aligned[i] or adx_aligned[i] <= 25)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0