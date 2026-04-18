#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) Breakout + Volume Spike + 1d EMA100 Trend Filter
# Uses daily EMA100 for trend filter, Donchian breakout for entry, volume spike for confirmation.
# Works in bull (breakout above upper band) and bear (breakdown below lower band) markets.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
name = "12h_Donchian20_VolumeSpike_1dEMA100"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA100
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 12h data (lookback 20 periods)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA100 on 1d data for trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate volume spike: current volume > 2.0 * 12-period average volume (6 days on 12h chart)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma_12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_100_1d_aligned[i]) or np.isnan(vol_ma_12[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = high_20[i]
        lower = low_20[i]
        ema_val = ema_100_1d_aligned[i]
        
        if position == 0:
            # Long: Break above upper band AND price above EMA100 AND volume spike
            if close_val > upper and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND price below EMA100 AND volume spike
            elif close_val < lower and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA100 (trend change) or touches lower band (mean reversion)
            if close_val < ema_val or close_val < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above EMA100 (trend change) or touches upper band (mean reversion)
            if close_val > ema_val or close_val > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals