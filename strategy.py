#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channel provides objective breakout levels based on price structure
# 1w EMA > 50-period ensures we trade only with the major trend, avoiding counter-trend whipsaws
# Volume spike confirms institutional participation behind the breakout
# Designed for very low frequency (50-150 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + price structure logic - EMA50 on weekly adapts to long-term trend

name = "6h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation (trend filter)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels from prior 20 periods (6h bars)
    # Upper = max(high of prior 20 bars), Lower = min(low of prior 20 bars)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need 1w EMA50 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_upper[i]  # Price breaks above upper Donchian
        breakout_short = close[i] < donchian_lower[i]  # Price breaks below lower Donchian
        
        # Trend filter: price above/below 1w EMA50 indicates trend direction
        # Long bias when price > 1w EMA50, short bias when price < 1w EMA50
        long_bias = close[i] > ema_50_1w_aligned[i]
        short_bias = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian with volume spike and long-term bullish bias
            if breakout_long and vol_spike and long_bias:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian with volume spike and long-term bearish bias
            elif breakout_short and vol_spike and short_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below lower Donchian or trend reversal (price < 1w EMA50)
            if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above upper Donchian or trend reversal (price > 1w EMA50)
            if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals