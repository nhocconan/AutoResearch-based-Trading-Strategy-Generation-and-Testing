#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 12h ATR-based volume confirmation and chop regime filter
    # Long when price breaks above Donchian upper band in choppy market (Chop > 61.8) with volume spike
    # Short when price breaks below Donchian lower band in choppy market (Chop > 61.8) with volume spike
    # Exit when price crosses opposite Donchian level OR market becomes trending (Chop < 38.2)
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 75-150 total trades over 4 years (~19-38/year)
    # Works in both bull and bear markets by using choppy market filter (Chop > 61.8)
    # Donchian channels provide clear breakout levels with built-in trend filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for price action (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for volume and chop confirmation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over 20 periods
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    
    # Lower band: lowest low over 20 periods
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 12h ATR (14-period) for volume normalization
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR (14-period) using Wilder's smoothing with min_periods via pandas ewm
    tr_series = pd.Series(tr)
    atr_12h = tr_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h Chop Index (Choppiness Index)
    # Sum of True Range over 14 periods with min_periods
    tr_series_for_sum = pd.Series(tr)
    sum_tr_14 = tr_series_for_sum.rolling(window=14, min_periods=14).sum().values
    
    # Chop Index = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    chop = np.where((atr_12h * 14) > 0,
                    100 * np.log10(sum_tr_14 / (atr_12h * 14)) / np.log10(14),
                    np.nan)
    
    # Align 12h Chop to 12h (wait for completed 12h bar)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate 12h volume average (20-period) with min_periods
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate 12h volume ATR ratio for volatility-adjusted volume confirmation
    vol_atr_ratio = volume_12h / (atr_12h + 1e-10)  # avoid division by zero
    vol_atr_ratio_series = pd.Series(vol_atr_ratio)
    vol_atr_ma_20 = vol_atr_ratio_series.rolling(window=20, min_periods=20).mean().values
    vol_atr_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_atr_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_atr_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h Chop > 61.8 (ranging/choppy market)
        ranging_market = chop_aligned[i] > 61.8
        # Exit regime: Chop < 38.2 (trending market begins)
        trending_market = chop_aligned[i] < 38.2
        
        # Volume confirmation: current 12h volume/ATR ratio > 1.3 * 20-period average
        vol_atr_current = vol_atr_ratio[i] if not np.isnan(vol_atr_ratio[i]) else 0
        volume_confirm = vol_atr_current > 1.3 * vol_atr_ma_aligned[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # Entry logic: Donchian breakout + volume confirmation + ranging market
        long_entry = long_breakout and volume_confirm and ranging_market
        short_entry = short_breakout and volume_confirm and ranging_market
        
        # Exit logic: price crosses opposite Donchian level OR market becomes trending
        long_exit = close[i] < donchian_lower_aligned[i] or trending_market
        short_exit = close[i] > donchian_upper_aligned[i] or trending_market
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_volume_atr_chop_v1"
timeframe = "4h"
leverage = 1.0