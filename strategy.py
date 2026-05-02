#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings below -80 or above -20
# with volume spike indicate potential reversal. 1d ADX > 25 ensures trades align with strong trend
# to avoid false signals in ranging markets. Designed for 50-150 total trades over 4 years (12-37/year)
# on 6h timeframe. Works in bull markets (buying oversold in uptrend) and bear markets
# (selling overbought in downtrend) by only taking trades in direction of 1d ADX.

name = "6h_WilliamsR_Extreme_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter (need +DI, -DI, DX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (using EMA as approximation with alpha=1/period)
    def wilders_smoothing(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    if len(tr) < period:
        return np.zeros(n)
    
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = williams_r(high, low, close, period=14)
    
    # Volume confirmation: 2.0x 20-period average (20*6h = 5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 30)  # 50 bars for Williams %R and ADX, 30 for volume
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R below -80 (oversold) with volume spike AND ADX > 25 (strong trend)
            if (wr[i] < -80 and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R above -20 (overbought) with volume spike AND ADX > 25 (strong trend)
            elif (wr[i] > -20 and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (exit oversold territory) OR ADX < 20 (trend weakening)
            if wr[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (exit overbought territory) OR ADX < 20 (trend weakening)
            if wr[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals