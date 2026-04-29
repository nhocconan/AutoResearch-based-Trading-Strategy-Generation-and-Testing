#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Donchian channels provide objective breakout levels that work in all market regimes
# Daily trend filter (price vs 50 EMA) ensures we trade breakouts in direction of higher timeframe trend
# Volume confirmation validates breakout strength and reduces false signals
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
# Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian Channel (20) on 4h data
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema50_1d = ema50_1d_aligned[i]
        
        # Determine trend regime from daily EMA50
        bullish_regime = curr_close > curr_ema50_1d
        bearish_regime = curr_close < curr_ema50_1d
        
        if position == 0:  # Flat - look for new entries
            # Look for Donchian breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian in bullish regime
                if bullish_regime and curr_close > curr_donchian_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian in bearish regime
                elif bearish_regime and curr_close < curr_donchian_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below Donchian middle OR opposite breakout occurs
            donchian_middle = (curr_donchian_upper + curr_donchian_lower) / 2
            if curr_close < donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above Donchian middle OR opposite breakout occurs
            donchian_middle = (curr_donchian_upper + curr_donchian_lower) / 2
            if curr_close > donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals