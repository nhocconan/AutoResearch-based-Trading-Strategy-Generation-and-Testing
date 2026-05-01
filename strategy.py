#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w ADX trend filter + ATR stoploss.
# Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Uses discrete sizing 0.30. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian calculated on completed 4h bar to avoid look-ahead. Volume spike filters low-momentum breakouts.
# ADX > 25 ensures trades only in established weekly trends (works in bull/bear markets).
# Target: 25-45 trades/year on 4h timeframe.

name = "4h_Donchian20_1dVolumeSpike_1wADX_Trend_v1"
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
    
    # Load 4h data ONCE before loop for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (no additional delay needed as based on completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Load 1d data ONCE before loop for volume spike filter (MTF)
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
    
    # Calculate 1w ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_first_w = np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])
    tr_w = np.concatenate([[tr_first_w], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean()
    atr_smooth_w = pd.Series(atr_w).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_smooth_w
    di_minus = 100 * dm_minus_smooth / atr_smooth_w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, volume, and ADX
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current 4h volume > 2.0x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 2.0)
        
        # Trend filter: 1w ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high AND volume spike AND strong trend
            if (curr_high > donchian_high_aligned[i] and 
                volume_spike and 
                strong_trend):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: Price breaks below Donchian low AND volume spike AND strong trend
            elif (curr_low < donchian_low_aligned[i] and 
                  volume_spike and 
                  strong_trend):
                signals[i] = -0.30
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
            # Exit: Price crosses below Donchian low OR volume dries up
            elif (curr_close < donchian_low_aligned[i]) or (not volume_spike):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Price crosses above Donchian high OR volume dries up
            elif (curr_close > donchian_high_aligned[i]) or (not volume_spike):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals