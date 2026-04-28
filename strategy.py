#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Trend Filter and Volume Spike
# Uses Bollinger Band width percentile to detect low volatility squeeze
# Breakout occurs when price closes outside BB(20,2) AND BB width > 20th percentile (expanding)
# Trend filter: 1d EMA50 slope > 0 for longs, < 0 for shorts
# Volume confirmation: current volume > 2.0x 20-bar average
# Designed for 6h timeframe to capture medium-term breakouts with low false signals
# Works in both bull and bear markets by trading breakouts in direction of 1d trend
# Target: 12-37 trades/year via strict squeeze + breakout + volume + trend confluence

name = "6h_BollingerSqueeze_Breakout_1dEMA50_Trend_VolumeSpike_v2"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA(50) and its slope
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA slope: current - previous (positive = rising trend)
    ema_50_slope = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    
    # Align 1d EMA slope to 6h timeframe
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    
    # Bollinger Bands (20, 2) on 6h data
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # BB Width percentile rank (lookback 50 bars) to detect squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for BB width percentile
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_slope_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        bb_width_pct = bb_width_percentile[i]
        ema_slope = ema_50_slope_aligned[i]
        price = close[i]
        
        # Breakout conditions
        breakout_up = price > bb_upper[i]
        breakout_down = price < bb_lower[i]
        
        # Squeeze condition: BB width at or below 20th percentile (low volatility)
        is_squeeze = bb_width_pct <= 0.20
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price above upper BB AND BB width expanding (>20th percentile)
            # AND 1d EMA50 slope positive (uptrend) AND volume confirmation
            if breakout_up and not is_squeeze and ema_slope > 0 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short breakout: price below lower BB AND BB width expanding (>20th percentile)
            # AND 1d EMA50 slope negative (downtrend) AND volume confirmation
            elif breakout_down and not is_squeeze and ema_slope < 0 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses below middle BB or trend fails
            if price < bb_middle[i] or ema_slope <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses above middle BB or trend fails
            if price > bb_middle[i] or ema_slope >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals