#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# Donchian breakouts capture volatility expansion after consolidation
# 12h EMA50 trend filter ensures we trade in direction of higher timeframe trend
# Volume confirmation validates breakout strength
# Works in bull markets (breakouts continue trend) and bear markets (breakouts reverse trend)
# Target: 20-50 trades/year (80-200 total over 4 years)

name = "4h_Donchian20_12hTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian Channel (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema50_12h = ema50_12h_aligned[i]
        
        # Determine trend regime from 12h EMA50
        bullish_regime = curr_close > curr_ema50_12h
        bearish_regime = curr_close < curr_ema50_12h
        
        if position == 0:  # Flat - look for new entries
            # Look for Donchian breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian in bullish regime
                if bullish_regime and curr_close > curr_donchian_upper:
                    signals[i] = 0.30
                    position = 1
                # Bearish breakout: price breaks below lower Donchian in bearish regime
                elif bearish_regime and curr_close < curr_donchian_lower:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below Donchian middle OR trend changes
            donchian_middle = (curr_donchian_upper + curr_donchian_lower) / 2
            if curr_close < donchian_middle or not bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above Donchian middle OR trend changes
            donchian_middle = (curr_donchian_upper + curr_donchian_lower) / 2
            if curr_close > donchian_middle or not bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals