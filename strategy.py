#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# Bollinger Band width < 20th percentile indicates low volatility (squeeze)
# Breakout occurs when price closes outside Bollinger Bands after squeeze
# 1w EMA50 provides higher timeframe trend bias to avoid counter-trend trades
# Volume confirmation (>1.5x average) filters weak breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_BollingerSqueeze_1wEMA50_Volume"
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
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_bb + (std_bb * bb_std)
    bb_lower = sma_bb - (std_bb * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Squeeze condition: BB width < 20th percentile
    squeeze = bb_width_percentile < 20
    
    # Breakout conditions
    breakout_up = close > bb_upper
    breakout_down = close < bb_lower
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(sma_bb[i]) or 
            np.isnan(std_bb[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: squeeze breakout up + above 1w EMA50 + volume confirmation
            if (squeeze[i-1] and breakout_up[i] and  # squeeze in previous bar, breakout now
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down + below 1w EMA50 + volume confirmation
            elif (squeeze[i-1] and breakout_down[i] and  # squeeze in previous bar, breakout now
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to middle Bollinger Band or breaks below 1w EMA50
            if (close[i] <= sma_bb[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to middle Bollinger Band or breaks above 1w EMA50
            if (close[i] >= sma_bb[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals