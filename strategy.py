#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_RegimeFilter
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakouts with 1-week EMA50 trend filter, volume spike confirmation (2.0x 20-period MA), and choppiness regime filter (CHOP < 38.2 = trending -> follow breakout, CHOP > 61.8 = range -> mean reversion at extremes). Uses ATR-based stoploss (2.5x). Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong moves in both bull and bear markets via regime-adaptive logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    R3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 2.0x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period)
    chop_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(chop_sum / range_hl) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA (50), volume MA (20), ATR (14), CHOP (14)
    start_idx = max(50, 20, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        chop_val = chop[i]
        
        # Regime filter: CHOP < 38.2 = trending (trend follow), CHOP > 61.8 = range (mean reversion)
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # In trending regime: follow trend with breakout
            # In ranging regime: mean reversion at extremes
            if is_trending:
                # Long: price breaks above R3 with volume confirmation and uptrend (price > 1w EMA50)
                long_signal = (high_val > R3_val) and (volume_val > 2.0 * vol_ma_val) and (close_val > ema_50_1w_val)
                # Short: price breaks below S3 with volume confirmation and downtrend (price < 1w EMA50)
                short_signal = (low_val < S3_val) and (volume_val > 2.0 * vol_ma_val) and (close_val < ema_50_1w_val)
            else:  # ranging regime
                # Long: price rejects below S3 (mean reversion up) with volume confirmation
                long_signal = (low_val < S3_val) and (close_val > S3_val) and (volume_val > 2.0 * vol_ma_val)
                # Short: price rejects above R3 (mean reversion down) with volume confirmation
                short_signal = (high_val > R3_val) and (close_val < R3_val) and (volume_val > 2.0 * vol_ma_val)
            
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
            # Hold long
            signals[i] = 0.25
            # Exit: ATR stoploss or trend reversal or ranging regime exit signal
            if (close_val < entry_price - 2.5 * atr_val or 
                close_val < ema_50_1w_val or
                (is_ranging and close_val < (R3_val + S3_val) / 2)):  # exit at midpoint in ranging
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR stoploss or trend reversal or ranging regime exit signal
            if (close_val > entry_price + 2.5 * atr_val or 
                close_val > ema_50_1w_val or
                (is_ranging and close_val > (R3_val + S3_val) / 2)):  # exit at midpoint in ranging
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0