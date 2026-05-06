#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike
# Long when price breaks above upper Donchian(20) AND 1w EMA34 uptrend AND volume > 1.5 * 20-bar avg volume
# Short when price breaks below lower Donchian(20) AND 1w EMA34 downtrend AND volume > 1.5 * 20-bar avg volume
# Exit with signal=0 when price reverses back inside Donchian(10) range (mean reversion)
# Uses discrete sizing 0.25 to balance opportunity and drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Donchian channels provide clear breakout levels; 1w EMA34 ensures higher-timeframe trend alignment
# Volume spike confirms institutional participation; works in bull via buying strength on upside breakouts
# and in bear via selling strength on downside breakdowns

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels from primary timeframe data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Upper and lower Donchian(20) for breakout signals
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Inner Donchian(10) for mean reversion exit
    upper_10 = high_series.rolling(window=10, min_periods=10).max().values
    lower_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(upper_10[i]) or np.isnan(lower_10[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: price breaks above upper Donchian(20) AND uptrend AND volume spike
            if close[i] > upper_20[i] and close[i] > ema_34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian(20) AND downtrend AND volume spike
            elif close[i] < lower_20[i] and close[i] < ema_34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverses back inside Donchian(10) range (mean reversion)
            if close[i] < upper_10[i] and close[i] > lower_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverses back inside Donchian(10) range (mean reversion)
            if close[i] < upper_10[i] and close[i] > lower_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals