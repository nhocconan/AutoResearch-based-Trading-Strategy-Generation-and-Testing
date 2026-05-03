#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets.
# Breakout direction aligned with weekly trend (price vs 1w EMA50) provides edge.
# Volume confirmation filters false breakouts. Designed for low trade frequency (12-37/year)
# on 6h timeframe to minimize fee drag while capturing explosive moves in any market regime.

name = "6h_BollingerSqueeze_1wEMA50_VolumeBreakout"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Bollinger Bands (20, 2) on 6h data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2.0 * std_20)
    lower_band = sma_20 - (2.0 * std_20)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized bandwidth
    
    # Bollinger Band squeeze: bandwidth < 20-period percentile 10 (low volatility)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).quantile(0.10).values
    squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(squeeze[i]) or np.isnan(volume_confirm[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Bollinger Band breakout above upper band in uptrend with volume confirmation
            if close[i] > upper_band[i] and close[i-1] <= upper_band[i-1] and is_uptrend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bollinger Band breakout below lower band in downtrend with volume confirmation
            elif close[i] < lower_band[i] and close[i-1] >= lower_band[i-1] and is_downtrend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band (SMA20) or reverses below lower band
            if close[i] < sma_20[i] and close[i-1] >= sma_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band (SMA20) or reverses above upper band
            if close[i] > sma_20[i] and close[i-1] <= sma_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals