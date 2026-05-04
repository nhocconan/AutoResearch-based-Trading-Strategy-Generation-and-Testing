#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Spike Confirmation
# Uses Bollinger Band width percentile to detect low volatility squeezes on 6h chart.
# Breakouts from squeezes are confirmed by 1d volume spike (volume > 1.5x 20-period EMA).
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets as volatility expansion precedes significant moves in any direction.

name = "6h_BollingerSqueeze_1dVolumeSpike_Breakout"
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
    
    # Get 1d data for volume spike calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume EMA20 for spike detection
    volume_ema20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume spike condition: volume > 1.5x EMA20
    volume_spike_1d = volume_1d > (1.5 * volume_ema20_1d)
    
    # Align volume spike to 6h timeframe (wait for completed 1d bar)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate Bollinger Bands on 6h (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    
    # Middle band (SMA)
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    
    # Standard deviation
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Upper and lower bands
    upper_band = sma + (bb_std * bb_stddev)
    lower_band = sma - (bb_std * bb_stddev)
    
    # Bollinger Band Width
    bb_width = (upper_band - lower_band) / sma
    
    # Bollinger Band Width percentile (50-period lookback) to detect squeeze
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Squeeze condition: BB width at or below 20th percentile (low volatility)
    squeeze_condition = bb_width_percentile <= 0.20
    
    # Breakout conditions
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: squeeze breakout to upside with volume confirmation
            if squeeze_condition[i-1] and breakout_up[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze breakout to downside with volume confirmation
            elif squeeze_condition[i-1] and breakout_down[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band or volatility expands significantly
            if close[i] <= sma[i] or bb_width_percentile[i] >= 0.80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band or volatility expands significantly
            if close[i] >= sma[i] or bb_width_percentile[i] >= 0.80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals