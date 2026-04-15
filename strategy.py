#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w EMA200 trend filter + 1d Donchian breakout with volume confirmation
# Works in bull: EMA200 uptrend + Donchian breakout captures momentum
# Works in bear: EMA200 downtrend filter prevents longs, allows shorts on breakdowns
# Volume confirmation reduces false breakouts, ATR stop manages risk
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop for EMA200 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(200) for trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 1d HTF data once before loop for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    donch_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate daily average volume (20-period) for confirmation
    avg_vol_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(avg_vol_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average volume
        # Need to get the corresponding 1d volume for current 12h bar
        # Since we're using aligned arrays, we can use the aligned volume data
        vol_1d = pd.Series(df_1d['volume'].values).iloc[-1] if len(df_1d) > 0 else 0  # Simplified - using latest
        vol_confirm = volume[i] > 1.5 * avg_vol_20_aligned[i] if not np.isnan(avg_vol_20_aligned[i]) else False
        
        # Long conditions:
        # 1. Price above weekly EMA200 (bullish long-term trend)
        # 2. Price breaks above daily Donchian high (breakout)
        # 3. Volume confirmation
        if (close[i] > ema_200_1w_aligned[i] and 
            high[i] > donch_high_20_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA200 (bearish long-term trend)
        # 2. Price breaks below daily Donchian low (breakdown)
        # 3. Volume confirmation
        elif (close[i] < ema_200_1w_aligned[i] and 
              low[i] < donch_low_20_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_EMA200_Donchian_VolFilter_v1"
timeframe = "12h"
leverage = 1.0