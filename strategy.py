#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above 6h Donchian upper channel AND 1d ADX > 25 AND volume > 1.5x 20-period volume median.
# Short when price breaks below 6h Donchian lower channel AND 1d ADX > 25 AND volume > 1.5x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian channels provide objective breakout levels; 1d ADX > 25 filters for strong trending markets on higher timeframe.
# Volume confirmation ensures breakout conviction. Works in bull markets (strong uptrend breakouts) and bear markets (strong downtrend breakdowns).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).

name = "6h_Donchian20_Breakout_1dADX_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 6h Donchian channels (20-period) using prior bar's data to avoid look-ahead
    # Upper channel = max(high) over past 20 periods (excluding current)
    # Lower channel = min(low) over past 20 periods (excluding current)
    high_past_20 = np.concatenate([[high[0]], high[:-1]])  # shift high by 1
    low_past_20 = np.concatenate([[low[0]], low[:-1]])    # shift low by 1
    donchian_upper = pd.Series(high_past_20).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_past_20).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX(14) trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range for ADX
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    tr1 = df_1d_high[1:] - df_1d_low[1:]
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr_first_1d = np.max([df_1d_high[0] - df_1d_low[0], np.abs(df_1d_high[0] - df_1d_close[0]), np.abs(df_1d_low[0] - df_1d_close[0])])
    tr_1d = np.concatenate([[tr_first_1d], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((df_1d_high[1:] - df_1d_high[:-1]) > (df_1d_low[:-1] - df_1d_low[1:]), 
                       np.maximum(df_1d_high[1:] - df_1d_high[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((df_1d_low[:-1] - df_1d_low[1:]) > (df_1d_high[1:] - df_1d_high[:-1]), 
                        np.maximum(df_1d_low[:-1] - df_1d_low[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smoothing(values, period):
        """Wilder's smoothing: first value is SMA, then recursive"""
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: SMA of first 'period' values
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, ADX, volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Donchian upper channel AND strong trend AND volume spike
            if curr_close > donchian_upper[i] and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Donchian lower channel AND strong trend AND volume spike
            elif curr_close < donchian_lower[i] and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower channel OR trend weakens (ADX < 20)
            elif curr_close < donchian_lower[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper channel OR trend weakens (ADX < 20)
            elif curr_close > donchian_upper[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals