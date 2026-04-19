#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# In high chop (range): mean revert at Donchian bands
# In low chop (trend): breakout in direction of trend
# Works in bull/bear by adapting to market regime. Target: 75-200 trades over 4 years.
name = "4h_Chop_Donchian_Volume_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index (calculated once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for daily
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ADX(14) components for directional movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR)/ (HHV-LLV)) / log10(period)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llvl = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hhvl - llvl + 1e-10)) / np.log10(14)
    chop = np.concatenate([[np.nan]*13, chop[13:]])  # align with index
    
    # Align daily indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for Donchian channels (calculated once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian(20) channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop > 61.8 = range (mean revert), chop < 38.2 = trend (breakout)
        in_range = chop_aligned[i] > 61.8
        in_trend = chop_aligned[i] < 38.2
        
        if position == 0:
            if in_range:
                # Mean reversion in range: fade at Donchian bands
                if close[i] <= donch_low[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donch_high[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            elif in_trend:
                # Breakout in trend: breakout in direction of ADX
                if plus_di_14[i] > minus_di_14[i] and close[i] > donch_high[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif minus_di_14[i] > plus_di_14[i] and close[i] < donch_low[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
                
        elif position == 1:
            # Long position: exit at opposite Donchian band or chop extreme
            if close[i] >= donch_high[i] or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit at opposite Donchian band or chop extreme
            if close[i] <= donch_low[i] or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals