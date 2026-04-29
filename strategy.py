#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction with 1d ADX regime filter and volume spike confirmation
# Uses 4h/1d for signal direction (trend + regime) and 1h only for precise entry timing
# Volume confirmation ensures breakouts have institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by only trading strong trends (ADX>25) with volume

name = "1h_Donchian20_4hDir_1dADX25_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[1:period+1])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smooth(dx, 14)
    
    # Align daily ADX to 1h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian(20) channels for directional bias
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_window = 20
    upper_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 4h trend direction: price relative to Donchian midpoint
    mid_4h = (upper_4h + lower_4h) / 2.0
    close_4h = df_4h['close'].values
    trend_4h = np.where(close_4h > mid_4h, 1, np.where(close_4h < mid_4h, -1, 0))
    
    # Align 4h trend and channels to 1h timeframe
    trend_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average on 1h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 35)  # warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if not in trading session or HTF data not available
        if not in_session[i] or np.isnan(adx_aligned[i]) or np.isnan(trend_aligned[i]) or \
           np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_adx = adx_aligned[i]
        curr_trend = trend_aligned[i]
        curr_upper = upper_aligned[i]
        curr_lower = lower_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        is_trending = curr_adx > 25
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation, in trending regime, and aligned with 4h trend
            if is_trending and curr_volume_confirm and curr_trend != 0:
                # Long when 4h trend is up and price breaks above 4h upper channel
                if curr_trend == 1 and curr_close > curr_upper:
                    signals[i] = 0.20
                    position = 1
                # Short when 4h trend is down and price breaks below 4h lower channel
                elif curr_trend == -1 and curr_close < curr_lower:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position - exit when 4h trend turns down or price hits lower channel
            if curr_trend == -1 or curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit when 4h trend turns up or price hits upper channel
            if curr_trend == 1 or curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals