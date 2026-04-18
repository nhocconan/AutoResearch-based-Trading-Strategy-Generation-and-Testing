#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band Squeeze with 1w Trend Filter and Volume Spike Confirmation
# Bollinger Band Squeeze: BB Width < 20th percentile indicates low volatility
# 1w EMA Trend Filter: Trade only in direction of weekly trend (EMA21)
# Volume Spike: Volume > 2x 20-period average confirms breakout conviction
# Works in bull markets (breakouts above upper BB) and bear markets (breakouts below lower BB)
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
name = "1d_BollingerSqueeze_1wEMA21_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA21 trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate BB Width percentile (20-day lookback) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1w EMA21 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_middle[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB Width < 20th percentile (low volatility)
        squeeze = bb_width_percentile[i] < 20
        
        # Trend filter: price above/below 1w EMA21
        uptrend = close[i] > ema21_1w_aligned[i]
        downtrend = close[i] < ema21_1w_aligned[i]
        
        if position == 0:
            # Long: squeeze + price breaks above upper BB + uptrend + volume spike
            if squeeze and close[i] > bb_upper[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze + price breaks below lower BB + downtrend + volume spike
            elif squeeze and close[i] < bb_lower[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle BB OR trend reverses
            if close[i] < bb_middle[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB OR trend reverses
            if close[i] > bb_middle[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals