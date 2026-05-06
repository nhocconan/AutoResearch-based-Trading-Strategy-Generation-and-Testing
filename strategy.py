#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation
# Long when Williams %R(14) < -80 (oversold) AND 12h close > 1d EMA50 AND volume > 1.5 * 20-bar average volume
# Short when Williams %R(14) > -20 (overbought) AND 12h close < 1d EMA50 AND volume > 1.5 * 20-bar average volume
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams %R identifies exhaustion points in both bull and bear markets
# 1d EMA50 filters for higher timeframe trend alignment (proven BTC/ETH edge from research)
# Volume spike confirmation reduces false signals during low participation
# Mean reversion works in ranging markets, trend filter avoids counter-trend trades

name = "12h_WilliamsR_MeanRev_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R(14) and 1d EMA50 ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 14 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14) * -100
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed bars)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: oversold AND uptrend AND volume spike
            if williams_r_aligned[i] < -80 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought AND downtrend AND volume spike
            elif williams_r_aligned[i] > -20 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (mean reversion complete)
            if williams_r_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (mean reversion complete)
            if williams_r_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals