#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter_v1
Hypothesis: Use 4h timeframe with Camarilla R3/S3 breakout confirmed by 1d EMA34 trend and volume spike. Add choppiness regime filter to avoid whipsaws in ranging markets. Camarilla levels provide intraday support/resistance that works in both bull and bear markets. Volume spike confirms breakout validity. Targets 20-50 trades/year to minimize fee drag. Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d OHLC for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla pivot levels: R3, S3 (most significant for breakouts)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness regime filter: avoid whipsaws in ranging markets
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # We only take breakout signals when CHOP < 61.8 (not strongly ranging)
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(tr[1:]).rolling(window=14, min_periods=14).sum().values  # exclude first NaN
    chop = 100 * np.log10(hl_range / true_range_sum) / np.log10(14)
    # Pad chop array to match length (first 13 values will be NaN due to rolling)
    chop_padded = np.concatenate([np.full(13, np.nan), chop])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(34, 20, 14, 14)  # EMA34, Donchian/volume avg, ATR, CHOP
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(chop_padded[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Only take signals when not strongly ranging (CHOP < 61.8)
            not_strongly_ranging = chop_padded[i] < 61.8
            
            # Long: break above Camarilla R3 + price above 1d EMA34 + volume spike
            long_entry = (close_val > camarilla_r3_aligned[i]) and \
                       (close_val > ema_34_aligned[i]) and \
                       volume_spike[i] and \
                       not_strongly_ranging
            # Short: break below Camarilla S3 + price below 1d EMA34 + volume spike
            short_entry = (close_val < camarilla_s3_aligned[i]) and \
                        (close_val < ema_34_aligned[i]) and \
                        volume_spike[i] and \
                        not_strongly_ranging
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Camarilla S3 break or ATR stoploss
            exit_condition = (close_val < camarilla_s3_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla R3 break or ATR stoploss
            exit_condition = (close_val > camarilla_r3_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0