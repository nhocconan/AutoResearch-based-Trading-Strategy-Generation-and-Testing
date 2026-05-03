#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation.
# Donchian breakout provides clear structure-based entries. 1d EMA50 filters for higher timeframe trend.
# Volume spike confirms institutional participation. Works in both bull and bear markets by only taking
# breakouts in the direction of the 1d trend. Target: 20-50 trades/year to minimize fee drag.

name = "4h_Donchian20_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get current values
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        donchian_high = highest_20[i]
        donchian_low = lowest_20[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(donchian_high) or np.isnan(donchian_low) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA50, bear if close < 1d EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Breakout conditions
        bull_breakout = high_val > donchian_high
        bear_breakout = low_val < donchian_low
        
        # Generate signals
        if position == 0:
            # Long: bullish breakout in bull regime with volume spike
            if is_bull_regime and bull_breakout and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout in bear regime with volume spike
            elif is_bear_regime and bear_breakout and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on bearish breakout or regime change to bear
            if bear_breakout or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on bullish breakout or regime change to bull
            if bull_breakout or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals