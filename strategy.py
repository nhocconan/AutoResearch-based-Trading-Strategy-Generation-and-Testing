#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) with uptrend (price > 1d EMA50) and volume > 2x 20-bar average.
# Short when Williams %R > -20 (overbought) with downtrend (price < 1d EMA50) and volume spike.
# Williams %R is a momentum oscillator that identifies overbought/oversold levels, effective in ranging markets.
# Combined with trend filter to avoid counter-trend trades and volume confirmation for conviction.
# Targets 50-150 trades over 4 years (12-37/year) with discrete position sizing (0.25).
# Works in both bull/bear markets by requiring 1d EMA50 trend alignment and volume confirmation.

name = "6h_WilliamsR_MeanRev_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if Williams %R not available
        if np.isnan(williams_r[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 1d EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend and curr_williams_r < -80 and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            elif is_downtrend and curr_williams_r > -20 and curr_volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum weakening) or volume drops
            if curr_williams_r > -50 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum weakening) or volume drops
            if curr_williams_r < -50 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals