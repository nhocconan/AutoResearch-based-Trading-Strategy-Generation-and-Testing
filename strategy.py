#!/usr/bin/env python3
"""
6h_WilliamsVixFix_v1
Hypothesis: Williams Vix Fix (WVF) identifies volatility spikes and market extremes on 6h timeframe.
- Long when WVF > 0.8 and price closes above 20-period EMA (fear exhaustion bounce)
- Short when WVF > 0.8 and price closes below 20-period EMA (fear continuation)
- Uses 12h EMA50 as trend filter to avoid counter-trend whipsaws
- Volume confirmation: require volume > 1.5x 20-period average
- Designed for low trade frequency (target: 12-30 trades/year) with clear volatility-based edge
- Works in both bull (catch panic bounces) and bear (catch fear continuations) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams Vix Fix: measures volatility spikes like VIX but for crypto
    # WVF = ((Highest Close in lookback - Low) / (Highest Close in lookback - Lowest Low in lookback)) * 100
    lookback = 22  # Similar to VIX calculation period
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    denominator = highest_close - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    wvf = ((highest_close - low) / denominator) * 100
    # Normalize to 0-1 range for easier thresholding
    wvf_normalized = wvf / 100.0
    
    # 20-period EMA for entry timing
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 22 for WVF, 20 for EMA/volume, 50 for 12h EMA)
    start_idx = max(22, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wvf_normalized[i]) or 
            np.isnan(ema20[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams Vix Fix extreme reading (>0.8 = high fear/volatility)
        wvf_extreme = wvf_normalized[i] > 0.8
        
        # Price relative to 20-period EMA
        price_above_ema = close[i] > ema20[i]
        price_below_ema = close[i] < ema20[i]
        
        # 12h trend filter
        trend_up = close[i] > ema50_12h_aligned[i]
        trend_down = close[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # Long: extreme fear + price above EMA20 + uptrend on 12h + volume spike
            if wvf_extreme and price_above_ema and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: extreme fear + price below EMA20 + downtrend on 12h + volume spike
            elif wvf_extreme and price_below_ema and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: fear subsides (WVF < 0.5) OR price falls below EMA20 OR trend turns down
            if (wvf_normalized[i] < 0.5) or (close[i] < ema20[i]) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: fear subsides (WVF < 0.5) OR price rises above EMA20 OR trend turns up
            if (wvf_normalized[i] < 0.5) or (close[i] > ema20[i]) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_v1"
timeframe = "6h"
leverage = 1.0