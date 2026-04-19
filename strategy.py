#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour timeframe strategy using weekly Bollinger Bands breakout
# with daily volume confirmation and weekly ADX trend filter.
# Weekly Bollinger Bands provide volatility-based support/resistance.
# Breakouts with volume and ADX>25 indicate strong momentum.
# Designed for low-frequency, high-conviction trades (target: 20-30 per year).
# Entry: Long when price breaks above upper BB with volume spike and ADX>25;
#        Short when price breaks below lower BB with volume spike and ADX>25.
# Exit: Return to middle BB or ADX<20.
# Uses strict conditions to limit trades and avoid overtrading.
name = "12h_BollingerBreakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Bollinger Bands (20, 2)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    bb_middle = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # Align weekly BB to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    # Daily ADX (14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((daily_high - np.roll(daily_high, 1)) > (np.roll(daily_low, 1) - daily_low),
                       np.maximum(daily_high - np.roll(daily_high, 1), 0), 0)
    dm_minus = np.where((np.roll(daily_low, 1) - daily_low) > (daily_high - np.roll(daily_high, 1)),
                        np.maximum(np.roll(daily_low, 1) - daily_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Daily volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper BB with volume and ADX>25
            if (close[i] > bb_upper_aligned[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below lower BB with volume and ADX>25
            elif (close[i] < bb_lower_aligned[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to middle BB or ADX<20
            if (close[i] < bb_middle_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to middle BB or ADX<20
            if (close[i] > bb_middle_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals