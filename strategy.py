#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h VWAP deviation with 1w EMA trend filter and volume surge confirmation
# Uses deviation from VWAP (volume-weighted average price) to identify mean reversion opportunities,
# confirmed by 1w EMA trend direction and volume > 2x average. Works in both bull and bear markets
# by taking mean-reversion trades against short-term deviations while following the weekly trend.
# Target: 25-40 trades/year to minimize fee decay while capturing mean reversion after volatility spikes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA on 1w (21-period)
    close_1w = df_1w['close'].values
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 21:
        ema_1w[20] = np.mean(close_1w[:21])
        for i in range(21, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 19) / 21
    
    # VWAP calculation for 4h (typical price * volume / cumulative volume)
    typical_price = (high + low + close) / 3
    vwap = np.full(n, np.nan)
    cum_vol = np.full(n, np.nan)
    cum_tpv = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            cum_vol[i] = volume[i]
            cum_tpv[i] = typical_price[i] * volume[i]
        else:
            cum_vol[i] = cum_vol[i-1] + volume[i]
            cum_tpv[i] = cum_tpv[i-1] + typical_price[i] * volume[i]
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] / cum_vol[i]
    
    # VWAP deviation percentage
    vwap_dev = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(vwap[i]) and vwap[i] > 0:
            vwap_dev[i] = (close[i] - vwap[i]) / vwap[i] * 100
    
    # 20-period average volume for surge detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align HTF indicators to 4h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 21)  # VWAP needs some history, EMA needs 21
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vwap_dev[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. VWAP deviation: > 1.5% away from VWAP (extended move)
        # 2. 1w EMA trend: price above EMA for long bias, below for short bias
        # 3. Volume confirmation: > 2x average volume (institutional interest)
        # 4. Mean reversion: price moving back toward VWAP
        extended_up = vwap_dev[i] > 1.5
        extended_down = vwap_dev[i] < -1.5
        uptrend = price > ema_1w_aligned[i]
        downtrend = price < ema_1w_aligned[i]
        volume_confirmation = vol_ratio > 2.0
        
        # Mean reversion signals: fade extended moves in direction of weekly trend
        if position == 0:
            # Long: fade downward extension in uptrend
            if extended_down and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: fade upward extension in downtrend
            elif extended_up and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to VWAP or trend changes
            if vwap_dev[i] < 0.2 or price < ema_1w_aligned[i]:  # near VWAP or trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to VWAP or trend changes
            if vwap_dev[i] > -0.2 or price > ema_1w_aligned[i]:  # near VWAP or trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_VWAPDeviation_1wEMA_Trend_VolumeSurge"
timeframe = "4h"
leverage = 1.0