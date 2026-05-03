#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d EMA34 up AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) low AND 1d EMA34 down AND volume > 1.5x 20-period MA.
# Exit when price crosses the opposite Donchian level (20-period low for longs, high for shorts).
# Uses 4h timeframe to target 75-200 trades over 4 years (19-50/year) with tight entry conditions.
# Donchian provides clear structure, 1d EMA34 filters for higher-timeframe trend, volume confirms participation.
# Designed to work in both bull (breakouts with trend) and bear (breakdowns with trend) markets.

name = "4h_Donchian20_1dEMA34_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 4h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # 1d EMA34 trend conditions
        ema_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
        ema_downtrend = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        
        if position == 0:
            # Long: price breaks above Donchian high AND EMA uptrend AND volume spike AND session
            if close[i] > donchian_high[i] and ema_uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA downtrend AND volume spike AND session
            elif close[i] < donchian_low[i] and ema_downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low (opposite level)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high (opposite level)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals