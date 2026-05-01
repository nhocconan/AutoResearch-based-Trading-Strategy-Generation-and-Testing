#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above Donchian upper band with volume > 2.0x 12h volume average and 1d ADX > 25 (trending).
# Short when price breaks below Donchian lower band with volume confirmation and 1d ADX > 25.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian bands calculated from prior completed 12h bar to avoid look-ahead.
# Volume spike and ADX filters ensure only strong breakouts in trending regimes.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 12-37 trades/year on 12h timeframe.

name = "12h_Donchian20_Breakout_1dADX_Volume_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for ADX and volume filters (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first_1d = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first_1d], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_first = np.maximum(high_1d[0] - high_1d[0], 0)  # 0
    dm_minus_first = np.maximum(low_1d[0] - low_1d[0], 0)   # 0
    dm_plus = np.concatenate([[dm_plus_first], dm_plus])
    dm_minus = np.concatenate([[dm_minus_first], dm_minus])
    
    # Smoothed TR, DM+
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(tr_14 > 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 > 0, (dm_minus_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 12h data ONCE before loop for Donchian bands and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, ADX, and Donchian
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.0x 12h volume average (using previous completed 12h bar)
        prev_12h_idx = i - 1
        if prev_12h_idx < 0:
            signals[i] = 0.0
            continue
            
        # Get 12h volume average for previous completed bar
        vol_lookback_start = max(0, prev_12h_idx - 19)
        vol_lookback_end = prev_12h_idx + 1
        if vol_lookback_end - vol_lookback_start < 20:
            signals[i] = 0.0
            continue
        vol_window = vol_12h[vol_lookback_start:vol_lookback_end]
        vol_ma_12h_prev = np.mean(vol_window) if len(vol_window) > 0 else 0
        
        if vol_ma_12h_prev <= 0:
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_12h_prev * 2.0)
        
        # Regime filter: 1d ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Calculate Donchian bands using previous completed 12h bar (lookback 20)
        lookback_start = max(0, prev_12h_idx - 19)
        lookback_end = prev_12h_idx + 1
        
        if lookback_end - lookback_start < 20:
            signals[i] = 0.0
            continue
            
        high_window = high_12h[lookback_start:lookback_end]
        low_window = low_12h[lookback_start:lookback_end]
        
        upper_band = np.max(high_window)
        lower_band = np.min(low_window)
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian upper band breakout up AND volume spike AND trending
            if (curr_high > upper_band and 
                volume_spike and 
                trending):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian lower band breakout down AND volume spike AND trending
            elif (curr_low < lower_band and 
                  volume_spike and 
                  trending):
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
            # Exit: price re-enters Donchian bands OR trend weakens (ADX < 20)
            elif (curr_low >= lower_band and curr_low <= upper_band) or \
                 (adx_aligned[i] < 20):
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
            # Exit: price re-enters Donchian bands OR trend weakens (ADX < 20)
            elif (curr_high >= lower_band and curr_high <= upper_band) or \
                 (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals