#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze + 1d Volume Spike + 6h Supertrend Trend Filter
# Uses Bollinger Band width contraction (squeeze) on 6h to identify low volatility periods,
# followed by expansion breakout confirmed by 1d volume spike (>2x 20-period average).
# 6h Supertrend (ATR=10, mult=3.0) filters breakout direction to avoid false signals.
# Designed for 12-30 trades/year (~50-120 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets: squeeze breakouts capture volatility expansion after consolidation,
# volume confirmation ensures institutional participation, Supertrend prevents counter-trend entries.

name = "6h_BBSqueeze_1dVolumeSpike_Supertrend"
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
    
    # Calculate 1d volume 20-period EMA for spike detection
    volume_ema20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * volume_ema20_1d)
    
    # Align volume spike to 6h timeframe (wait for completed 1d bar)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate 6h Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    bb_width = upper_band - lower_band
    
    # Bollinger Band Squeeze: width below 50-period percentile of width (low volatility)
    bb_width_ma50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < (0.8 * bb_width_ma50)  # 20% below average width
    
    # Breakout condition: price closes outside Bollinger Bands
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Calculate 6h Supertrend (ATR=10, mult=3.0) for trend filter
    atr_period = 10
    atr_mult = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (EMA with alpha=1/period)
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])  # Seed with SMA
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band_st = hl2 + (atr_mult * atr)
    lower_band_st = hl2 - (atr_mult * atr)
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period-1] = upper_band_st[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band_st[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band_st[i], supertrend[i-1])
            direction[i] = -1
    
    # Align Supertrend direction (already on 6h, use completed bar)
    supertrend_direction = np.roll(direction, 1)
    supertrend_direction[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(supertrend_direction[i]) or 
            np.isnan(squeeze_condition[i]) or 
            np.isnan(breakout_up[i]) or 
            np.isnan(breakout_down[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: squeeze breakout up + volume spike + Supertrend uptrend
            if (squeeze_condition[i-1] and breakout_up[i] and 
                volume_spike_1d_aligned[i] > 0.5 and 
                supertrend_direction[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze breakout down + volume spike + Supertrend downtrend
            elif (squeeze_condition[i-1] and breakout_down[i] and 
                  volume_spike_1d_aligned[i] > 0.5 and 
                  supertrend_direction[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Bollinger Bands OR Supertrend turns down
            if (close[i] >= lower_band[i] and close[i] <= upper_band[i]) or supertrend_direction[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Bollinger Bands OR Supertrend turns up
            if (close[i] >= lower_band[i] and close[i] <= upper_band[i]) or supertrend_direction[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals