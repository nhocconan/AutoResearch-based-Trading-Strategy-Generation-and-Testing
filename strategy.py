#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# In bull regime (price > 1d EMA50), we go long on upper Donchian breakout with volume spike.
# In bear regime (price < 1d EMA50), we go short on lower Donchian breakout with volume spike.
# Exit on opposite Donchian breakout or loss of volume confirmation. This captures trends
# while avoiding whipsaws in ranging markets by requiring both breakout and volume confirmation.

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
    
    # Calculate 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        dc_high = high_20[i]
        dc_low = low_20[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(dc_high) or np.isnan(dc_low) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA50, bear if close < 1d EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Generate signals
        if position == 0:
            # Look for breakout entries with volume spike
            if is_bull_regime and high_val > dc_high and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_regime and low_val < dc_low and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long: exit on lower Donchian breakout or loss of volume confirmation
            if low_val < dc_low or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: exit on upper Donchian breakout or loss of volume confirmation
            if high_val > dc_high or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals