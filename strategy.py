#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter.
# Long when price breaks above Donchian upper channel AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Short when price breaks below Donchian lower channel AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian calculated on completed 4h bar to avoid look-ahead. Volume spike filters low-momentum breakouts.
# ADX > 25 ensures trades only in established weekly trends (works in both bull and bear markets).
# Target: 20-50 trades/year on 4h timeframe.

name = "4h_Donchian20_Breakout_1dVolumeSpike_1wADX_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 4h data ONCE before loop for Donchian channels (primary timeframe data)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper: 20-period rolling max of high
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: 20-period rolling min of low
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (no additional delay needed for Donchian as it's based on completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Load 1d data ONCE before loop for volume filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data ONCE before loop for ADX trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_first_w = np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])
    tr_w = np.concatenate([[tr_first_w], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    
    # Directional Movement
    dm_plus_w = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                         np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus_w = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                          np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    # First values
    dm_plus_w = np.concatenate([[0], dm_plus_w])
    dm_minus_w = np.concatenate([[0], dm_minus_w])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        result[period-1] = np.nansum(source[:period])
        for i in range(period, len(source)):
            result[i] = result[i-1] - (result[i-1] / period) + source[i]
        return result
    
    atr_w = wilders_smoothing(tr_w, 14)
    dm_plus_w_smooth = wilders_smoothing(dm_plus_w, 14)
    dm_minus_w_smooth = wilders_smoothing(dm_minus_w, 14)
    
    # DI+ and DI-
    di_plus_w = np.where(atr_w != 0, (dm_plus_w_smooth / atr_w) * 100, 0)
    di_minus_w = np.where(atr_w != 0, (dm_minus_w_smooth / atr_w) * 100, 0)
    
    # DX and ADX
    dx_w = np.where((di_plus_w + di_minus_w) != 0, 
                    np.abs(di_plus_w - di_minus_w) / (di_plus_w + di_minus_w) * 100, 0)
    adx_w = np.full_like(dx_w, np.nan, dtype=float)
    adx_w[13:] = pd.Series(dx_w).rolling(window=14, min_periods=14).mean().values[13:]
    
    # Align ADX to 4h timeframe
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, volume, and ADX
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.0x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 2.0)
        
        # Trend filter: 1w ADX > 25
        strong_trend = adx_w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper AND volume spike AND strong trend
            if (curr_close > donchian_upper_aligned[i] and 
                volume_spike and 
                strong_trend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower AND volume spike AND strong trend
            elif (curr_close < donchian_lower_aligned[i] and 
                  volume_spike and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian lower (reversal signal)
            elif curr_close < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian upper (reversal signal)
            elif curr_close > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals