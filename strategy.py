#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1d chop regime filter
# - Long when price breaks above 20-period 12h Donchian upper band with volume > 1.5x 20-bar avg AND 1d chop < 61.8 (trending)
# - Short when price breaks below 20-period 12h Donchian lower band with volume > 1.5x 20-bar avg AND 1d chop < 61.8 (trending)
# - Exit when price crosses 12h EMA(20) in opposite direction
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong trends, chop filter avoids whipsaws in ranging markets
# - Volume confirmation ensures breakout validity

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian upper and lower bands
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe (completed 12h bar only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Pre-compute 12h EMA(20) for exit signal
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 1d Chopiness Index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max/min high/low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Chopiness Index: 100 * log10(sum(atr14) / (max_high - min_low)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop_denominator = max_high_14 - min_low_14
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_1d = np.where(chop_denominator > 0, chop_raw, 50.0)  # Default to 50 when denominator <= 0
    
    # Align Chopiness Index to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Define trending regime: chop < 61.8
    trending_regime = chop_1d_aligned < 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian upper with volume spike and trending regime
            if (prices['close'].iloc[i] > donchian_upper_aligned[i] and 
                vol_spike.iloc[i] and 
                trending_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian lower with volume spike and trending regime
            elif (prices['close'].iloc[i] < donchian_lower_aligned[i] and 
                  vol_spike.iloc[i] and 
                  trending_regime[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price crosses 12h EMA(20) in opposite direction
            if position == 1 and prices['close'].iloc[i] < ema_20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > ema_20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals