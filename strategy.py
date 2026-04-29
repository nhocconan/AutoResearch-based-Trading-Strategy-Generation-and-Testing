#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with Weekly EMA50 trend filter and Volume confirmation
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# Weekly EMA50 ensures we trade breakouts in direction of higher timeframe trend
# Volume confirmation validates breakout strength and reduces false signals
# Works in all regimes: breakouts after consolidation phases
# Target: 15-30 trades/year (60-120 total over 4 years)

name = "1d_Donchian20_WeeklyTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian Channel (20) on 1d data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema50_1w = ema50_1w_aligned[i]
        
        # Determine trend regime from weekly EMA50
        bullish_regime = curr_close > curr_ema50_1w
        bearish_regime = curr_close < curr_ema50_1w
        
        if position == 0:  # Flat - look for new entries
            # Look for Donchian breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian in bullish regime
                if bullish_regime and curr_high > highest_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian in bearish regime
                elif bearish_regime and curr_low < lowest_20[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if curr_close < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if curr_close > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals