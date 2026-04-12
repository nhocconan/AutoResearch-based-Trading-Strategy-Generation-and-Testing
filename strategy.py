#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla breakout with volume confirmation and ADX trend filter
# Uses 1d Camarilla levels (H4/L4) for breakout signals, volume filter to avoid false breakouts,
# and ADX to ensure trades are taken only in trending markets. Designed to work in both bull and bear
# markets by only trading in the direction of the trend. Targets 20-40 trades/year to minimize fee drag.

name = "4h_1d_camarilla_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar data to avoid look-ahead
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate 1d Camarilla levels (H4/L4 breakout)
    pivot_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d_prev = high_1d_prev - low_1d_prev
    h4_prev = pivot_prev + (range_1d_prev * 1.1 / 2)
    l4_prev = pivot_prev - (range_1d_prev * 1.1 / 2)
    
    # Align to 4h
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_prev)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_prev)
    
    # Volume filter: 20-period average on 4h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # ADX trend filter: 14-period on 4h
    # Calculate +DM, -DM, TR
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    up_move = high_series.diff()
    down_move = -low_series.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high_series - low_series
    tr2 = np.abs(high_series - close_series.shift(1))
    tr3 = np.abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    tr_ma = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_ma = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_ma = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    adx_values = adx.values
    # Trend strong when ADX > 25
    trend_strong = adx_values > 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any values not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(trend_strong[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above H4 with volume and strong trend
        long_signal = close[i] > h4_aligned[i] and volume_ok[i] and trend_strong[i]
        # Short: break below L4 with volume and strong trend
        short_signal = close[i] < l4_aligned[i] and volume_ok[i] and trend_strong[i]
        
        # Exit on opposite breakout (mean reversion to L4/H4 or opposite signal)
        exit_long = close[i] < l4_aligned[i]  # Exit long if price breaks below L4
        exit_short = close[i] > h4_aligned[i]  # Exit short if price breaks above H4
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals