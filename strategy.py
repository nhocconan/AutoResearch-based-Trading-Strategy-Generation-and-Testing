#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter for 1h entries
# Uses 4h Donchian channels (20) for breakout direction, 1d EMA(50) for trend filter,
# and volume spike (1.5x 20-period average) for confirmation. Restricts entries to 08-20 UTC
# to avoid low-liquidity periods. Position size fixed at 0.20 to manage risk.
# Designed for low trade frequency (15-37/year) to minimize fee drift while capturing
# trending moves in both bull and bear markets via breakout logic with confirmation.

name = "1h_donchian_breakout_volume_1dtrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h (with shift(1) for completed bars only)
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Long breakout: price > 4h Donchian high + volume spike + above 1d EMA50
        if (close[i] > highest_20_aligned[i] and vol_spike[i] and 
            close[i] > ema_50_aligned[i]):
            signals[i] = 0.20
        
        # Short breakout: price < 4h Donchian low + volume spike + below 1d EMA50
        elif (close[i] < lowest_20_aligned[i] and vol_spike[i] and 
              close[i] < ema_50_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0
    
    return signals