#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily trend filter and volume confirmation
# Uses 12h Donchian channel (20-period) for breakout signals
# Daily EMA50 ensures we only trade breakouts in direction of daily trend
# Volume spike (>2.0x 20-period average) confirms momentum
# Works in both bull and bear markets by aligning with higher timeframe trend
# Target: 12-25 trades/year (50-100 total over 4 years)

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 12h timeframe (completed 1d bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema50 = ema_50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine daily trend: price above/below EMA50
        uptrend = curr_close > curr_ema50
        downtrend = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in direction of daily trend
            if curr_volume_confirm:
                # Bullish breakout: price breaks above Donchian high in uptrend
                if uptrend and curr_close > curr_donchian_high:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Donchian low in downtrend
                elif downtrend and curr_close < curr_donchian_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to Donchian low OR breaks below with volume
            if curr_close <= curr_donchian_low or (curr_close < curr_donchian_low and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to Donchian high OR breaks above with volume
            if curr_close >= curr_donchian_high or (curr_close > curr_donchian_high and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals