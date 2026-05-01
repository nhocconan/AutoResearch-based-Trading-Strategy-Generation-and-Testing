#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d volume spike and 1h EMA50 trend filter
# Williams %R identifies overbought/oversold conditions; extreme readings (<-90 or >-10) signal exhaustion
# Volume spike confirms institutional participation in the reversal
# 1h EMA50 ensures we trade in the direction of the intermediate trend to avoid counter-trend whipsaws
# Designed for low trade frequency: ~20-30 trades/year per symbol with 0.25 sizing
# Works in bull/bear: EMA filter adapts to trend direction, volume confirms reversal validity

name = "4h_WilliamsR_ExtremeRev_1dVolumeSpike_1hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1h HTF data for EMA50 trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 60:
        return np.zeros(n)
    
    # Williams %R(14) from 4h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    williams_r = -100 * (highest_high - close) / hl_range
    
    # 1d volume spike: volume > 2.5 * 20-period EMA
    vol_1d = df_1d['volume'].values
    vol_1d_series = pd.Series(vol_1d)
    vol_ema_20_1d = vol_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.5 * vol_ema_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1h EMA50 for trend filter
    close_1h = df_1h['close'].values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 55, 65)  # Need Williams %R, 1d volume, 1h EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1h_aligned[i]) or np.isnan(vol_ema_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = close[i] > ema_50_1h_aligned[i]
        short_bias = close[i] < ema_50_1h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation from 1d
            if volume_spike_1d_aligned[i]:
                # Long: Williams %R < -90 (oversold) with long bias
                if williams_r[i] < -90 and long_bias:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -10 (overbought) with short bias
                elif williams_r[i] > -10 and short_bias:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Wait for volume confirmation
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion) or overbought
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion) or oversold
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals