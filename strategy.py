#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND 1w ADX > 25 AND volume > 1.5x 20-period average volume.
# Short when price breaks below Donchian(20) lower band AND 1w ADX > 25 AND volume confirmation.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian calculated from prior completed 1d bar to avoid look-ahead.
# ADX filter ensures trades only in established weekly trends (avoids choppy markets).
# Volume spike filters low-momentum breakouts. Works in bull (breakouts with uptrend) and bear (breakouts with downtrend).
# Target: 15-35 trades/year on 1d timeframe.

name = "1d_Donchian20_1wADX_Volume_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Donchian and volume filters (primary timeframe data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate Donchian Channel (20-period)
    # Upper band: 20-period high
    # Lower band: 20-period low
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (no additional delay needed for Donchian as it's based on completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d volume average (20-period)
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_w = pd.Series(tr_w).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth_w = pd.Series(dm_plus_w).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth_w = pd.Series(dm_minus_w).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus_w = 100 * dm_plus_smooth_w / atr_w
    di_minus_w = 100 * dm_minus_smooth_w / atr_w
    
    # DX and ADX
    dx_w = 100 * np.abs(di_plus_w - di_minus_w) / (di_plus_w + di_minus_w)
    # Handle division by zero
    dx_w = np.where((di_plus_w + di_minus_w) == 0, 0, dx_w)
    adx_w = pd.Series(dx_w).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 1d timeframe (no additional delay needed for ADX as it's based on completed 1w bar)
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
        
        # Volume spike: current volume > 1.5x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        
        # Trend filter: 1w ADX > 25 indicates strong trend
        strong_trend = adx_w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band AND volume spike AND strong trend
            if (curr_close > donchian_upper_aligned[i] and 
                volume_spike and 
                strong_trend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower band AND volume spike AND strong trend
            elif (curr_close < donchian_lower_aligned[i] and 
                  volume_spike and 
                  strong_trend):
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
            # Exit: price breaks below Donchian lower band (contrarian exit) OR ADX weakens
            elif (curr_close < donchian_lower_aligned[i]) or (adx_w_aligned[i] < 20):
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
            # Exit: price breaks above Donchian upper band (contrarian exit) OR ADX weakens
            elif (curr_close > donchian_upper_aligned[i]) or (adx_w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals