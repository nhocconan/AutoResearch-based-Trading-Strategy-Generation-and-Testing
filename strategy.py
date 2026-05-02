#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume spike and 1d trend filter
# Bollinger Band width < 20th percentile identifies low volatility squeeze
# Breakout occurs when price closes outside BB(20,2) with volume > 2x 20-period EMA
# 1d EMA50 filter ensures trades align with daily trend to avoid counter-trend entries
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Works in bull markets (breakout above upper BB + price > 1d EMA50) and bear markets (breakdown below lower BB + price < 1d EMA50)
# Uses discrete position sizing (0.25) to balance return potential with drawdown control

name = "6h_BollingerSqueeze_Breakout_12hVolume_1dEMA50"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h volume EMA20 for spike detection
    vol_ema_20_12h = pd.Series(df_12h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20_12h)
    
    # Bollinger Bands (6h timeframe)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Bollinger Band Width percentile (20-period lookback)
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger squeeze condition: BB width in lowest 20th percentile
        squeeze_condition = bb_width_percentile[i] <= 0.20
        
        # Breakout conditions
        breakout_up = close[i] > upper_bb[i]
        breakout_down = close[i] < lower_bb[i]
        
        # Volume confirmation: current 6h volume > 2x 12h volume EMA20
        # Approximate 12h volume EMA for 6bar by using aligned value (conservative)
        volume_confirmation = volume[i] > (2.0 * vol_ema_20_12h_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            if squeeze_condition and breakout_up and volume_confirmation and close[i] > ema_50_1d_aligned[i]:
                # Long: Bollinger squeeze breakout up with volume and daily trend alignment
                signals[i] = 0.25
                position = 1
            elif squeeze_condition and breakout_down and volume_confirmation and close[i] < ema_50_1d_aligned[i]:
                # Short: Bollinger squeeze breakdown down with volume and daily trend alignment
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price re-enters Bollinger Bands OR daily trend turns bearish
            if close[i] < sma_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price re-enters Bollinger Bands OR daily trend turns bullish
            if close[i] > sma_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals