#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h ADX trend filter
# Donchian channels identify structural breakouts; volume confirms conviction; 
# 12h ADX > 25 ensures trades only in trending markets to avoid false breakouts in ranging conditions.
# Works in bull markets (buying upper band breakouts in uptrend) and bear markets
# (selling lower band breakdowns in downtrend) by only taking trades in direction of 12h ADX.
# Designed for 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Donchian20_Breakout_Volume_12hADX25_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h ADX for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                               np.abs(low_12h[1:] - close_12h[:-1])))
    
    # Handle first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    plus_di = 100 * wilder_smooth(plus_dm, period) / wilder_smooth(tr, period)
    minus_di = 100 * wilder_smooth(minus_dm, period) / wilder_smooth(tr, period)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(donchian_window, 30)  # 20 bars for Donchian, 30 for ADX
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian with volume spike AND ADX > 25 (trending market)
            if (close[i] > upper_channel[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian with volume spike AND ADX > 25 (trending market)
            elif (close[i] < lower_channel[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower Donchian (trend reversal) OR ADX < 20 (trend weakening)
            if close[i] < lower_channel[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian (trend reversal) OR ADX < 20 (trend weakening)
            if close[i] > upper_channel[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals