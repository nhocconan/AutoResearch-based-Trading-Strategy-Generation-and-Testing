#!/usr/bin/env python3
# Hypothesis: 4h Williams %R reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. ADX(14) > 25 confirms trending market.
# Long when: Williams %R crosses above -80 (from oversold), ADX > 25, volume spike
# Short when: Williams %R crosses below -20 (from overbought), ADX > 25, volume spike
# Exit when: Williams %R crosses above -20 (long) or below -80 (short) OR ADX < 20
# Position size: 0.25 to limit drawdown. Target: 20-40 trades/year.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets.

name = "4h_WilliamsR_ADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    def williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.fillna(-50).values  # neutral when undefined
    
    wr = williams_r(high, low, close, 14)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_series(raw, period):
        smoothed = np.zeros_like(raw)
        smoothed[period-1] = np.mean(raw[:period])
        for i in range(period, len(raw)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + raw[i]
        return smoothed
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    # ADX
    adx = smooth_series(dx, 14)
    adx = np.where(np.isnan(adx), 0, adx)
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 (from oversold), ADX > 25, volume spike
            if (wr[i] > -80 and wr[i-1] <= -80 and 
                adx_aligned[i] > 25 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 (from overbought), ADX > 25, volume spike
            elif (wr[i] < -20 and wr[i-1] >= -20 and 
                  adx_aligned[i] > 25 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR ADX < 20 (trend weak)
            if (wr[i] > -20 and wr[i-1] <= -20) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR ADX < 20 (trend weak)
            if (wr[i] < -80 and wr[i-1] >= -80) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals