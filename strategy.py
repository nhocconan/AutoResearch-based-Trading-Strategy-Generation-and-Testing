#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion_1dTrendFilter
Hypothesis: Williams Vix Fix identifies volatility spikes and mean reversion opportunities. 
In ranging markets (ADX < 25 on 1d), extreme Vix Fix readings (>0.8) signal imminent reversals. 
Volume confirmation filters false signals. Works in both bull/bear by fading volatility extremes 
during low-trend regimes. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend/regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ADX on 1d for regime filtering (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    # Align ADX to 6h with 1-bar delay (completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    
    # Calculate Williams Vix Fix on 6h data
    # Vix Fix = ((Highest Close in Period - Low) / Highest Close in Period) * 100
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    vix_fix = ((highest_close - low) / highest_close) * 100
    # Normalize to 0-1 range (typical Vix Fix ranges 0-100)
    vix_fix_norm = vix_fix / 100
    
    # Volume confirmation: 1.3x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for highest_close and ADX
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(vix_fix_norm[i]) or
            np.isnan(highest_close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade in ranging markets (ADX < 25 = low trend)
        ranging_market = adx_aligned[i] < 25
        
        if position == 0 and ranging_market:
            # Look for mean reversion signals from volatility extremes
            # Long: Vix Fix > 0.8 (extreme fear) + volume spike
            # Short: Vix Fix < 0.2 (extreme complacency) + volume spike
            long_signal = (vix_fix_norm[i] > 0.8) and volume_spike[i]
            short_signal = (vix_fix_norm[i] < 0.2) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when Vix Fix returns to normal levels (< 0.5) or trend emerges
            exit_signal = (vix_fix_norm[i] < 0.5) or (adx_aligned[i] >= 25)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Vix Fix returns to normal levels (> 0.5) or trend emerges
            exit_signal = (vix_fix_norm[i] > 0.5) or (adx_aligned[i] >= 25)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion_1dTrendFilter"
timeframe = "6h"
leverage = 1.0