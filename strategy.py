#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 4h timeframe, Donchian(20) breakouts with 1d EMA34 trend filter and volume confirmation capture strong directional moves in both bull and bear markets. The 4h timeframe balances trade frequency and signal quality, while the 1d EMA34 filter ensures we trade with the higher timeframe trend. Volume confirmation adds validity to breakouts. Targets 19-50 trades/year to stay within fee drag limits and achieve Sharpe > 0 on BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels on 4h (20-period)
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period = ~3.33 days on 4h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(20, 34)  # Donchian lookback, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_1d_val = ema_34_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = vol_val > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper with uptrend (close > EMA34) and volume confirmation
            long_signal = (high_val > upper) and (close_val > ema_34_1d_val) and volume_confirmed
            # Short: price breaks below Donchian lower with downtrend (close < EMA34) and volume confirmation
            short_signal = (low_val < lower) and (close_val < ema_34_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below Donchian lower (exit long)
            if low_val < lower:
                signals[i] = 0.0
                position = 0
            # 2. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above Donchian upper (exit short)
            if high_val > upper:
                signals[i] = 0.0
                position = 0
            # 2. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0