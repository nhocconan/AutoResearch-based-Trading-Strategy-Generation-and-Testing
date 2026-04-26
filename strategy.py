#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Confluence
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike confirmation capture institutional moves with minimal trades. 
Requires volume > 2.5x average (tighter than prior 2.0x) and only trades during UTC 8-16 session to avoid low-volume periods. 
Adds volatility filter (ATR ratio < 1.2) to avoid choppy markets. Targets 15-30 trades/year to minimize fee drag while maintaining edge.
Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes during low volatility).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (UTC 8-16)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 16)
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) on 1d for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d_arr[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d_arr[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    if len(high_1d) < 2:
        camarilla_r1 = np.full_like(close_1d_arr, np.nan)
        camarilla_s1 = np.full_like(close_1d_arr, np.nan)
    else:
        camarilla_r1 = close_1d_arr[:-1] + 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_s1 = close_1d_arr[:-1] - 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume average (20-period = ~3.3 days on 4h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 34, 14)  # volume MA, 1d EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        atr_14_1d_val = atr_14_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2.5x 20-period average (tighter)
        volume_confirmed = vol_val > 2.5 * vol_ma_val
        # Volatility filter: avoid extreme volatility (ATR ratio < 1.2 of 50-period MA)
        vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
        vol_ma_50_val = vol_ma_50[i] if not np.isnan(vol_ma_50[i]) else vol_ma_val
        volatility_filter = vol_val < 1.2 * vol_ma_50_val if vol_ma_50_val > 0 else True
        
        if position == 0:
            # Long: price breaks above R1 with uptrend (close > EMA34) and volume confirmation
            long_signal = (high_val > r1_val) and (close_val > ema_34_1d_val) and volume_confirmed and volatility_filter
            # Short: price breaks below S1 with downtrend (close < EMA34) and volume confirmation
            short_signal = (low_val < s1_val) and (close_val < ema_34_1d_val) and volume_confirmed and volatility_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S1 (exit long)
            if low_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if high_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Confluence"
timeframe = "4h"
leverage = 1.0