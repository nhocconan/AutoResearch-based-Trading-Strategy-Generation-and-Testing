#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND close > 1w EMA50 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below Donchian lower band AND close < 1w EMA50 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price crosses the Donchian middle band (mean reversion to median)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Donchian channels provide clear structure; 1w EMA50 ensures alignment with weekly trend
# Volume spike confirms institutional breakout validity; middle-band exit works in ranging markets

name = "1d_Donchian20_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: break above upper band AND uptrend AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle band (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle band (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals