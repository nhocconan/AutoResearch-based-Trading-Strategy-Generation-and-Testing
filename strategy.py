#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1w EMA50 ensures alignment with weekly trend.
# Volume confirmation filters false breakouts. Designed for low frequency (~10-25 trades/year)
# to minimize fee drag and work in both bull and bear markets by following the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
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
    
    # Get 1w data for EMA50 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d Donchian channels (20-period) using previous day's data to avoid look-ahead
    # We need 20 periods of high/low, so we shift by 1 to use only completed data
    high_shifted = pd.Series(high).shift(1)
    low_shifted = pd.Series(low).shift(1)
    donchian_high = high_shifted.rolling(window=20, min_periods=20).max().values
    donchian_low = low_shifted.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume confirmation (volume > 1.5 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high in uptrend with volume confirmation
            if high[i] > donchian_high[i] and is_uptrend and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low in downtrend with volume confirmation
            elif low[i] < donchian_low[i] and is_downtrend and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low (reversal) or time-based exit (optional)
            if low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high (reversal) or time-based exit
            if high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals