#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Uses Camarilla pivot levels (R3, S3) from 1d timeframe for breakout entries, filtered by 1d EMA34 trend and volume spike (>2x average). Adds choppiness regime filter (CHOP > 61.8 = range = avoid breakouts). Enters long when price breaks above 1d R3 AND 1d close > 1d EMA34 (uptrend) AND volume > 2x average AND choppy regime filter passed. Enters short when price breaks below 1d S3 AND 1d close < 1d EMA34 (downtrend) AND volume > 2x average AND choppy regime filter passed. Exits when price reverts to 1d close (mean reversion) OR trend breaks. Uses 4h timeframe with tight entries to avoid fee drag: target 25-40 trades/year. Camarilla R3/S3 levels provide higher-probability breakout points than R1/S1, volume confirmation avoids low-conviction moves, and chop filter prevents whipsaws in ranging markets. Works in both bull and bear markets via 1d trend filter and volume confirmation.
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
    
    # Get 1d data for Camarilla pivot calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    hl_range_1d = df_1d['high'].values - df_1d['low'].values
    camarilla_r3_1d = df_1d['close'].values + hl_range_1d * 1.1 / 4
    camarilla_s3_1d = df_1d['close'].values - hl_range_1d * 1.1 / 4
    
    # Align 1d indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter: avoid breakouts in choppy markets (CHOP > 61.8 = range)
    # CHOP = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: use rolling max/min and ATR
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[0], tr1])  # align length
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 * 14 / np.log10(14) / (max_high - min_low + 1e-10))
    chop_regime = chop > 61.8  # True = choppy/range, avoid breakouts
    chop_filter = ~chop_regime  # Only allow breakouts in trending markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), chop (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_1d_val = camarilla_r3_1d_aligned[i]
        s3_1d_val = camarilla_s3_1d_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        close_1d_val = close_1d_aligned[i]
        
        if position == 0:
            # Look for entry: price breakout above R3 (long) or below S3 (short) with trend, volume, and regime filter
            # Long: price > R3 AND 1d uptrend AND volume confirmation AND not choppy
            long_condition = (close_val > r3_1d_val) and (close_val > ema_1d_val) and vol_conf and chop_ok
            # Short: price < S3 AND 1d downtrend AND volume confirmation AND not choppy
            short_condition = (close_val < s3_1d_val) and (close_val < ema_1d_val) and vol_conf and chop_ok
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to 1d close (mean reversion) OR trend breaks
            exit_condition = (close_val <= close_1d_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to 1d close (mean reversion) OR trend breaks
            exit_condition = (close_val >= close_1d_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0