#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout + 1d EMA50 trend filter + volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# 1d EMA50 ensures we trade with higher timeframe trend to avoid whipsaws
# Volume spike (>1.5x 20-period EMA) confirms breakout authenticity
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# Works in ranging markets (squeeze detection) and trending markets (breakout continuation)

name = "6h_BollingerSqueeze_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands on 6h: 20-period SMA, 2 standard deviations
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = (upper_band - lower_band) / sma_20  # Normalized bandwidth
    
    # Bollinger Band squeeze: bandwidth below 20-period EMA of bandwidth
    bb_width_ema = pd.Series(bb_width).ewm(span=20, adjust=False, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ema
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Bollinger Band breakout with 1d trend filter
        # Long: squeeze breakout above upper band + price above 1d EMA50 + volume spike
        # Short: squeeze breakout below lower band + price below 1d EMA50 + volume spike
        if position == 0:
            if (squeeze[i-1] and not squeeze[i] and  # Squeeze release
                close[i] > upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            elif (squeeze[i-1] and not squeeze[i] and  # Squeeze release
                  close[i] < lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below 20-period SMA OR squeeze re-establishes
            if close[i] < sma_20[i] or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above 20-period SMA OR squeeze re-establishes
            if close[i] > sma_20[i] or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals