#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Donchian breakout captures momentum, EMA50 trend filter ensures we trade with higher timeframe direction,
# volume spike confirms institutional participation. This structure works in both bull and bear markets
# by only taking breakouts in the direction of the 1d trend, reducing false signals and overtrading.

name = "4h_Donchian20_1dTrend_VolumeSpike_Regime"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        close_val = close[i]
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
        
        # Breakout conditions with volume confirmation
        long_breakout = close_val > donchian_high
        short_breakout = close_val < donchian_low
        
        # Regime-filtered entry conditions
        if position == 0:
            # Long: bullish breakout in bull regime with volume spike
            if is_bull_regime and long_breakout and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout in bear regime with volume spike
            elif is_bear_regime and short_breakout and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below Donchian low or regime change to bear
            if close_val < donchian_low or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above Donchian high or regime change to bull
            if close_val > donchian_high or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals