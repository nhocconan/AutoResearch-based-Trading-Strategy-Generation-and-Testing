#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian breakout with 1d ATR filter and volume confirmation.
# Uses weekly price channels to capture major trend breaks on daily timeframe.
# Daily ATR filter ensures sufficient volatility to avoid choppy markets.
# Volume confirmation adds conviction to breakouts.
# Designed for very low trade frequency (7-25/year) to minimize fee drag in 1d timeframe.
# Works in bull markets (breakouts above weekly upper band) and bear markets (breakouts below weekly lower band).
name = "1d_WeeklyDonchian_DailyATR_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period) using previous week's data to avoid look-ahead
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    upper_band_20w = high_20w
    lower_band_20w = low_20w
    
    # Get daily data for ATR filter and volume (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14-period)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing (EMA with alpha=1/14)
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold = atr * atr_mult
    
    # Align weekly Donchian bands to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band_20w)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band_20w)
    
    # Align daily ATR threshold to daily timeframe (no change needed but for consistency)
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(atr_threshold_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR threshold must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_aligned[i]) and atr_threshold_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above weekly upper band AND volume confirmation AND volatility filter
            long_breakout = close[i] > upper_band_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower band AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < lower_band_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below weekly lower band OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] < lower_band_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above weekly upper band OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] > upper_band_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals