#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Spike and 12h Trend Filter
# Uses Bollinger Bands (20,2) on 6h to detect low volatility squeeze (BBWidth < 20th percentile).
# Breakout triggers when price closes outside bands AFTER a squeeze.
# Confirmed by 1d volume spike (>1.5x 20-period average) and 12h EMA50 trend filter.
# Designed for low-frequency, high-conviction trades (~15-35/year) to minimize fee drag.
# Works in bull/bear markets: squeeze breakouts capture volatility expansion after consolidation,
# volume confirms institutional participation, 12h EMA filters counter-trend noise.

name = "6h_BollingerSqueeze_1dVolumeSpike_12hEMA50_TrendFilter"
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
    
    # Get 1d data for volume spike - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    
    # Align 1d volume spike to 6h timeframe (wait for completed 1d bar)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Get 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Bollinger Bands on 6h (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized width
    
    # Calculate 50th percentile of BBWidth for squeeze threshold (using expanding window)
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(20, len(bb_width)):
        bb_width_percentile[i] = np.percentile(bb_width[20:i+1], 20)  # 20th percentile
    
    squeeze_condition = bb_width < bb_width_percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for Bollinger Band breakout AFTER squeeze
        bullish_breakout = close[i] > upper_band[i] and squeeze_condition[i-1]
        bearish_breakout = close[i] < lower_band[i] and squeeze_condition[i-1]
        
        if position == 0:
            # Long conditions: bullish breakout + volume spike + price above 12h EMA50
            if (bullish_breakout and 
                volume_spike_1d_aligned[i] > 0.5 and  # Boolean treated as 0/1
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish breakout + volume spike + price below 12h EMA50
            elif (bearish_breakout and 
                  volume_spike_1d_aligned[i] > 0.5 and
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Bollinger Bands OR 12h EMA50 turns bearish
            if (close[i] >= lower_band[i] and close[i] <= upper_band[i]) or \
               (i > 0 and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Bollinger Bands OR 12h EMA50 turns bullish
            if (close[i] >= lower_band[i] and close[i] <= upper_band[i]) or \
               (i > 0 and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals