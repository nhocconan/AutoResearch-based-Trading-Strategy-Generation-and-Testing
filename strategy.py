#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high AND 1w close > 1w EMA50 (uptrend) AND volume > 2.0 * 20-day avg volume
# Short when price breaks below 20-day low AND 1w close < 1w EMA50 (downtrend) AND volume > 2.0 * 20-day avg volume
# Exit when price retraces to the 10-day EMA (mean reversion to intermediate trend)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w EMA50 provides strong trend filter between 1d and 1w for better regime adaptation
# Volume threshold increased to 2.0x to reduce false breakouts and lower trade frequency
# 10-day EMA exit works in ranging markets and captures mean reversion after breakout failure
# This strategy focuses on BTC and ETH as primary targets, avoiding SOL-only bias

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
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
    
    # Calculate Donchian channels for 1d timeframe (based on previous 20 bars)
    # Upper band = 20-day high
    # Lower band = 20-day low
    # Exit band = 10-day EMA
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    exit_ema = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Shift by 1 to use only completed bar data (no look-ahead)
    donchian_upper_prev = np.roll(donchian_upper, 1)
    donchian_lower_prev = np.roll(donchian_lower, 1)
    exit_ema_prev = np.roll(exit_ema, 1)
    donchian_upper_prev[0] = np.nan
    donchian_lower_prev[0] = np.nan
    exit_ema_prev[0] = np.nan
    
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
    
    # Calculate volume confirmation: volume > 2.0 * 20-day average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_prev[i]) or np.isnan(donchian_lower_prev[i]) or 
            np.isnan(exit_ema_prev[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper band AND uptrend AND volume spike
            if close[i] > donchian_upper_prev[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower_prev[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to 10-day EMA (mean reversion)
            if close[i] <= exit_ema_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to 10-day EMA (mean reversion)
            if close[i] >= exit_ema_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals