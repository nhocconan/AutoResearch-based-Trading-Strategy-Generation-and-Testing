#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour ADX trend filter and volume confirmation.
# Uses 4-hour Donchian(20) channels for breakout signals.
# Filters: 12-hour ADX > 25 for trending markets, volume > 1.5x 20-period average.
# Designed to capture strong trending moves with low turnover (target: 20-50 trades/year).
# Works in bull markets (breakout momentum) and bear markets (trend following via ADX).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian(20) channels (upper/lower)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donch_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    donch_upper_4h = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_4h = align_htf_to_ltf(prices, df_4h, donch_lower)
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14)
    plus_dm = np.diff(high_12h, prepend=high_12h[0])
    minus_dm = np.diff(low_12h, prepend=low_12h[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high_12h, prepend=high_12h[0]))
    tr2 = np.abs(np.diff(low_12h, prepend=low_12h[0]))
    tr3 = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_12h + 1e-10)
    minus_di_12h = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_12h + 1e-10)
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_12h_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_4h[i]) or 
            np.isnan(donch_lower_4h[i]) or 
            np.isnan(adx_12h_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance signal quality)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: 12h ADX > 25 for trending markets
        trending = adx_12h_4h[i] > 25
        
        # Price relative to Donchian channels
        price_above_upper = close[i] > donch_upper_4h[i]
        price_below_lower = close[i] < donch_lower_4h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume and trending
            if (price_above_upper and volume_filter and trending):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume and trending
            elif (price_below_lower and volume_filter and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower OR ADX < 20 (trend weakening)
            if (close[i] < donch_lower_4h[i]) or (adx_12h_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper OR ADX < 20 (trend weakening)
            if (close[i] > donch_upper_4h[i]) or (adx_12h_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hADX25_Volume"
timeframe = "4h"
leverage = 1.0