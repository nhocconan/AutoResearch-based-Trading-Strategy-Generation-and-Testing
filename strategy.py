#!/usr/bin/env python3
"""
1h_MultiTimeframe_Momentum_v1
Hypothesis: Use 4h Donchian breakout for trend direction and 1d RSI for momentum confirmation, with 1h price action for entry timing. 
Go long when price breaks above 4h Donchian upper band AND 1d RSI > 55, short when price breaks below 4h Donchian lower band AND 1d RSI < 45. 
Requires volume > 1.5x 20-period average for confirmation. Uses session filter (08-20 UTC) to avoid low-liquidity hours. 
Target: 15-30 trades/year by combining multiple filters to reduce noise. Works in bull markets via trend following and in bear via short signals.
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    donch_len = 20
    upper_4h = np.full_like(high_4h, np.nan)
    lower_4h = np.full_like(low_4h, np.nan)
    
    if len(high_4h) >= donch_len:
        for i in range(donch_len, len(high_4h)):
            upper_4h[i] = np.max(high_4h[i-donch_len:i])
            lower_4h[i] = np.min(low_4h[i-donch_len:i])
    
    # Align Donchian channels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    rsi_period = 14
    rsi_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, rsi_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and in_session:
            # Long: price breaks above 4h Donchian upper + RSI > 55 + volume
            if close[i] > upper_4h_aligned[i] and rsi_1d_aligned[i] > 55 and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower + RSI < 45 + volume
            elif close[i] < lower_4h_aligned[i] and rsi_1d_aligned[i] < 45 and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h Donchian lower OR RSI < 40
            if close[i] < lower_4h_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above 4h Donchian upper OR RSI > 60
            if close[i] > upper_4h_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_MultiTimeframe_Momentum_v1"
timeframe = "1h"
leverage = 1.0