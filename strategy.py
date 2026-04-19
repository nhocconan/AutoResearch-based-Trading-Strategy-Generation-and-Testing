#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze + 1d volume confirmation + 1d trend filter
# - Bollinger Band width (BBW) < 20th percentile indicates squeeze (low volatility)
# - Breakout from squeeze with volume > 1.5x 20-period 1d average volume for confirmation
# - 1d EMA(50) trend filter: long when price > EMA50, short when price < EMA50
# - Works in both bull and bear markets by capturing volatility expansion after consolidation
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "6h_BollingerSqueeze_1dVolume_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_width = bb_up - bb_low
    
    # Bollinger Band width percentile (20-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BBW < 20th percentile
    squeeze = bb_width_percentile < 20
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 6h: 1d has 4x 6h bars, so divide by 4
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 4.0)
        
        if position == 0:
            # Look for breakout from squeeze with volume and trend alignment
            if (squeeze_aligned[i] and 
                volume_filter and 
                close[i] > bb_up[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (squeeze_aligned[i] and 
                  volume_filter and 
                  close[i] < bb_low[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to middle band or trend reversal
            if close[i] < bb_mid[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to middle band or trend reversal
            if close[i] > bb_mid[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals