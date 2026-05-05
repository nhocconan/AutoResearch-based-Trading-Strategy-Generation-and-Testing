#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND price > 1w EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile (squeeze) AND price < 1w EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price returns to BB middle (20 SMA) OR BB width > 50th percentile (squeeze ends)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 30-100 total trades over 4 years (7-25/year)
# Bollinger squeeze identifies low volatility breakout setups; 1w EMA200 filters primary trend; volume confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1d_BBSqueeze_1wEMA200_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Bollinger Bands
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile (20-day lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Get 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need enough for EMA200
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or
            np.isnan(bb_width_pct[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB, BB width in squeeze (<20th percentile), above 1w EMA200, volume confirmation
            if (close[i] > bb_upper[i] and bb_width_pct[i] < 0.2 and 
                close[i] > ema200_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB, BB width in squeeze (<20th percentile), below 1w EMA200, volume confirmation
            elif (close[i] < bb_lower[i] and bb_width_pct[i] < 0.2 and 
                  close[i] < ema200_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price returns to middle BB OR squeeze ends (width > 50th percentile)
            if close[i] < bb_middle[i] or bb_width_pct[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price returns to middle BB OR squeeze ends (width > 50th percentile)
            if close[i] > bb_middle[i] or bb_width_pct[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals