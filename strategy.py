#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze + Donchian Breakout + 1d ADX Trend Filter
# Bollinger Band Width < 20th percentile indicates low volatility squeeze
# Breakout occurs when price crosses Donchian(20) channel with volume > 1.5x average
# 1d ADX > 25 filters for trending markets only to avoid false breakouts in ranging markets
# Works in both bull and bear markets by trading breakouts in the direction of the higher-timeframe trend
# Discrete sizing (0.25) limits fee drag and controls drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Bollinger Squeeze identifies low volatility precursors to explosive moves
# Donchian breakouts capture the ensuing trend with objective entry/exit levels
# ADX filter ensures we only trade when there is sufficient trend strength to follow through

name = "6h_BB_Squeeze_Donchian_Breakout_1dADX25_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX on 1d data for trend strength filter
    # ADX calculation requires +DI, -DI, and TR
    up_move = df_1d['high'].diff()
    down_move = df_1d['low'].diff().multiply(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(plus_dm) >= period:
        plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / wilders_smoothing(tr, period)
        minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / wilders_smoothing(tr, period)
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = wilders_smoothing(dx_1d, period)
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # Bollinger Band Width on 6h for squeeze detection
    bb_period = 20
    bb_std = 2.0
    if len(close) >= bb_period:
        bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
        bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
        bb_upper = bb_ma + (bb_std_dev * bb_std)
        bb_lower = bb_ma - (bb_std_dev * bb_std)
        bb_width = (bb_upper - bb_lower) / bb_ma
        
        # Calculate 20th percentile of BB Width for squeeze threshold (using expanding window)
        bb_width_percentile = np.full(n, np.nan)
        for i in range(bb_period, n):
            if i >= 20:  # Need minimum lookback for percentile
                bb_width_percentile[i] = np.percentile(bb_width[bb_period:i+1], 20)
    else:
        bb_width = np.full(n, np.nan)
        bb_width_percentile = np.full(n, np.nan)
    
    # Donchian Channel on 6h for breakout detection
    dc_period = 20
    if len(high) >= dc_period and len(low) >= dc_period:
        donchian_high = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
        donchian_low = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for Bollinger Band Squeeze (width below 20th percentile)
        is_squeeze = bb_width[i] < bb_width_percentile[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high during squeeze with volume and ADX filter
            if (is_squeeze and 
                close[i] > donchian_high[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low during squeeze with volume and ADX filter
            elif (is_squeeze and 
                  close[i] < donchian_low[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR ADX falls below 20 (trend weakening)
            if close[i] < donchian_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR ADX falls below 20 (trend weakening)
            if close[i] > donchian_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals