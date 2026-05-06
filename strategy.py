#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d trend filter and volume spike
# Long when price breaks above R4 (strong breakout level) AND 1d close > 1d open (bullish daily candle) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below S4 (strong breakdown level) AND 1d close < 1d open (bearish daily candle) AND volume > 2.0 * 20-bar avg volume
# Exit with signal=0 when price reverses back inside the Camarilla H3-L3 range (mean reversion)
# Uses discrete sizing 0.25 to balance opportunity and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla R4/S4 are extreme breakout levels that require strong momentum, reducing false breakouts
# 1d candle direction ensures alignment with higher-timeframe sentiment to avoid counter-trend trades
# Volume spike confirms institutional participation and reduces whipsaws
# Works in bull via buying strength on upside breakouts, works in bear via selling strength on downside breakdowns

name = "6h_Camarilla_R4S4_1dCandleDir_VolumeSpike_v1"
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
    
    # Get 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate 1d candle direction: 1 for bullish (close > open), -1 for bearish (close < open), 0 for doji
    candle_dir_1d = np.where(close_1d > open_1d, 1, np.where(close_1d < open_1d, -1, 0))
    
    # Calculate Camarilla levels from daily data
    # Camarilla formula: H-L range based
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 6
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    candle_dir_1d_aligned = align_htf_to_ltf(prices, df_1d, candle_dir_1d)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(candle_dir_1d_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R4 (strong resistance) AND bullish daily candle AND volume spike
            if close[i] > R4_aligned[i] and candle_dir_1d_aligned[i] > 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 (strong support) AND bearish daily candle AND volume spike
            elif close[i] < S4_aligned[i] and candle_dir_1d_aligned[i] < 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverses back inside H3-L3 range (mean reversion)
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverses back inside H3-L3 range (mean reversion)
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals