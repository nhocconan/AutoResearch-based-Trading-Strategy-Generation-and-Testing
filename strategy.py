#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze + Volume Spike + 12h EMA50 Trend Filter
# Bollinger Band Squeeze (low volatility) precedes explosive moves in both bull and bear markets.
# Entry: BB Width < 20th percentile (squeeze) + Volume Spike (>2.0x 20-bar avg) + price breaks above/below BB (±2σ)
# Direction: 12h EMA50 trend filter (long if price > EMA50, short if price < EMA50)
# Exit: BB Width > 50th percentile (squeeze end) or opposite BB touch
# Works in bull markets via breakouts and in bear markets via volatility expansion shorts
# Target: 12-35 trades/year via strict squeeze + volume + trend confluence

name = "6h_BB_Squeeze_VolumeSpike_12hEMA50_Trend_v1"
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
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = upper_band - lower_band
    
    # BB Width percentile (20-day lookback for squeeze detection)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, bb_period, 20, 100)  # Need sufficient history
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Squeeze condition: BB Width < 20th percentile (low volatility)
        squeeze = bb_width_percentile[i] < 20.0
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: squeeze + volume spike + breakout up + uptrend
            if squeeze and vol_spike and breakout_up and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: squeeze + volume spike + breakout down + downtrend
            elif squeeze and vol_spike and breakout_down and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on squeeze end (BB Width > 50th percentile) or opposite band touch
            if bb_width_percentile[i] > 50.0 or close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on squeeze end (BB Width > 50th percentile) or opposite band touch
            if bb_width_percentile[i] > 50.0 or close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals