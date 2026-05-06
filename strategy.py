#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume spike
# Long when Williams %R crosses above -80 (oversold recovery) AND 1d close > 1d EMA50 (uptrend) AND volume > 1.8 * 20-bar avg volume
# Short when Williams %R crosses below -20 (overbought rejection) AND 1d close < 1d EMA50 (downtrend) AND volume > 1.8 * 20-bar avg volume
# Exit when Williams %R returns to -50 (mean reversion equilibrium) or opposing signal appears
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R is effective in ranging and trending markets, capturing reversals at extremes
# 1d EMA50 provides robust trend filter for better regime adaptation in both bull and bear markets
# Volume threshold set to 1.8x to reduce false signals while maintaining sufficient trade frequency

name = "4h_WilliamsR_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R for 4h timeframe (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams %R reversal signals with trend and volume filters
            # Long: Williams %R crosses above -80 (from below) AND uptrend AND volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND downtrend AND volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or crosses below -80 (re-rejection)
            if williams_r[i] >= -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or crosses above -20 (re-rejection)
            if williams_r[i] <= -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals