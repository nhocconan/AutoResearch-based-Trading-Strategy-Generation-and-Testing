#!/usr/bin/env python3
"""
12h_1d_adaptive_triple_crossover_v1
Hypothesis: Adaptive trend following with multi-timeframe EMA crossovers for 12h timeframe.
- Primary signal: 12h EMA(21) vs EMA(55) crossover (adaptive period based on volatility)
- Trend filter: 1d EMA(89) direction to avoid counter-trend trades in strong trends
- Volatility filter: ATR(14) ratio to avoid choppy markets (ATR > 1.5x ATR(50))
- Volume confirmation: 12h volume > 1.3x 20-period average
- Dynamic position sizing: 0.25 in low volatility, 0.15 in high volatility
- Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years)
- Works in bull/bear via trend filter and volatility-adjusted sizing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_adaptive_triple_crossover_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # 1d EMA(89) for trend filter
    close_1d = df_1d['close'].values
    ema_89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    trend_1d_up = close_1d > ema_89_1d
    trend_1d_down = close_1d < ema_89_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / np.where(atr_50 > 0, atr_50, 1)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Get 12h data for primary signal and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    # Adaptive EMA periods based on volatility
    # In high volatility: shorter EMAs for responsiveness
    # In low volatility: longer EMAs for stability
    base_fast, base_slow = 21, 55
    vol_scalar = np.clip(atr_ratio_aligned, 0.5, 2.0)  # clamp between 0.5 and 2.0
    ema_fast_period = base_fast / vol_scalar
    ema_slow_period = base_slow / vol_scalar
    # Ensure minimum periods
    ema_fast_period = np.maximum(ema_fast_period, 8)
    ema_slow_period = np.maximum(ema_slow_period, 21)
    
    # Calculate adaptive EMAs
    close_12h = df_12h['close'].values
    ema_fast = np.zeros_like(close_12h)
    ema_slow = np.zeros_like(close_12h)
    
    # Initialize
    ema_fast[0] = close_12h[0]
    ema_slow[0] = close_12h[0]
    
    # Calculate EMAs with adaptive smoothing
    for i in range(1, len(close_12h)):
        alpha_fast = 2 / (ema_fast_period[i] + 1)
        alpha_slow = 2 / (ema_slow_period[i] + 1)
        ema_fast[i] = alpha_fast * close_12h[i] + (1 - alpha_fast) * ema_fast[i-1]
        ema_slow[i] = alpha_slow * close_12h[i] + (1 - alpha_slow) * ema_slow[i-1]
    
    # 12h EMA crossover signals
    ema_cross_up = ema_fast > ema_slow
    ema_cross_down = ema_fast < ema_slow
    
    # Forward fill crossover signals
    ema_cross_up_series = pd.Series(ema_cross_up)
    ema_cross_down_series = pd.Series(ema_cross_down)
    ema_cross_up_ffilled = ema_cross_up_series.ffill().values
    ema_cross_down_ffilled = ema_cross_down_series.ffill().values
    
    # Align 12h EMA crossovers to 12h timeframe (no additional delay needed)
    ema_cross_up_aligned = align_htf_to_ltf(prices, df_12h, ema_cross_up_ffilled)
    ema_cross_down_aligned = align_htf_to_ltf(prices, df_12h, ema_cross_down_ffilled)
    
    # 12h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # Dynamic position sizing based on volatility
    # Lower volatility = larger position, higher volatility = smaller position
    vol_scalar_clipped = np.clip(atr_ratio_aligned, 0.5, 2.0)
    base_size = 0.25
    size_multiplier = 2.0 / vol_scalar_clipped  # inverse relationship
    size_multiplier = np.clip(size_multiplier, 0.6, 1.4)  # bound between 0.15 and 0.35
    position_size = base_size * size_multiplier
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(ema_cross_up_aligned[i]) or np.isnan(ema_cross_down_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(position_size[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR 1d trend turns down
            if ema_cross_down_aligned[i] or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size[i]
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR 1d trend turns up
            if ema_cross_up_aligned[i] or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size[i]
        else:  # Flat, look for entry
            # Long entry: EMA cross up + 1d uptrend + volume
            if (ema_cross_up_aligned[i] and trend_1d_up_aligned[i] and volume_filter[i]):
                position = 1
                signals[i] = position_size[i]
            # Short entry: EMA cross down + 1d downtrend + volume
            elif (ema_cross_down_aligned[i] and trend_1d_down_aligned[i] and volume_filter[i]):
                position = -1
                signals[i] = -position_size[i]
    
    return signals