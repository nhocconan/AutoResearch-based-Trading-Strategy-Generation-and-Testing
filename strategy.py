#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion + 1d volume spike + 6h ADX regime filter
# - Williams %R(14) calculated on 12h timeframe for overbought/oversold conditions
# - Long when %R < -80 (oversold) with 1d volume > 1.5x 20-bar average AND 6h ADX < 25 (ranging market)
# - Short when %R > -20 (overbought) with 1d volume > 1.5x 20-bar average AND 6h ADX < 25 (ranging market)
# - Exit when %R returns to -50 (mean reversion target)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Williams %R works well in ranging markets (2022-2025) with mean reversion tendency
# - Volume confirmation ensures participation during reversals
# - ADX filter avoids trending markets where mean reversion fails

name = "12h_1d_6h_williamsr_meanrev_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    df_6h = get_htf_data(prices, '6h')
    if len(df_12h) < 50 or len(df_1d) < 50 or len(df_6h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators for Williams %R calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 12h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14) * -100
    
    # Align Williams %R to 12h timeframe (completed 12h bar only)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 6h ADX(14) for regime filter
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_6h[1:] - low_6h[:-1])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement (+DM and -DM)
    up_move = high_6h[1:] - high_6h[:-1]
    down_move = low_6h[:-1] - low_6h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if period < len(data):
            result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_6h = wilder_smooth(tr, 14)
    plus_di_6h = 100 * wilder_smooth(plus_dm, 14) / atr_6h
    minus_di_6h = 100 * wilder_smooth(minus_dm, 14) / atr_6h
    dx_6h = 100 * np.abs(plus_di_6h - minus_di_6h) / (plus_di_6h + minus_di_6h)
    adx_6h = wilder_smooth(dx_6h, 14)
    
    # Align ADX to 12h timeframe (completed 6h bar only)
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_12h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(adx_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R oversold (< -80) with volume spike and ranging market (ADX < 25)
            if (williams_r_12h_aligned[i] < -80 and 
                vol_spike_1d_aligned[i] and 
                adx_6h_aligned[i] < 25):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R overbought (> -20) with volume spike and ranging market (ADX < 25)
            elif (williams_r_12h_aligned[i] > -20 and 
                  vol_spike_1d_aligned[i] and 
                  adx_6h_aligned[i] < 25):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when Williams %R returns to -50 (mean reversion target)
            if position == 1 and williams_r_12h_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r_12h_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals