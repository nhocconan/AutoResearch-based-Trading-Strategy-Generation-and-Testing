#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 in bull regime (price > 1d EMA34), short when breaks below S3 in bear regime (price < 1d EMA34).
Volume confirmation: volume > 1.5 * ATR(14) to avoid false breakouts.
Only trade in direction of 1d trend to avoid counter-trend whipsaws.
Discrete sizing: 0.25 to minimize fee churn. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend regime (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar: based on previous 4h bar's OHLC
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe (they are already 4h-aligned)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate ATR for volume spike filter (using 4h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for 1d EMA34 (34) and ATR (14)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Determine 1d trend regime
        # Bull regime: price > EMA34
        # Bear regime: price < EMA34
        # Range regime: near EMA34 (within 0.5*ATR)
        regime_threshold = 0.5 * atr[i]
        
        if close[i] > ema_34_1d_aligned[i] + regime_threshold:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1d_aligned[i] - regime_threshold:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades
        
        # Volume spike: current volume > 1.5 * ATR (adaptive threshold)
        volume_spike = volume[i] > 1.5 * atr[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R3 AND volume spike AND bull regime
            long_setup = (close[i] > camarilla_r3_aligned[i]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below Camarilla S3 AND volume spike AND bear regime
            short_setup = (close[i] < camarilla_s3_aligned[i]) and volume_spike and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price closes below Camarilla R3 OR regime turns bearish OR max holding period (12 bars = 2 days)
            if (close[i] < camarilla_r3_aligned[i]) or (regime == 'bear') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price closes above Camarilla S3 OR regime turns bullish OR max holding period (12 bars = 2 days)
            if (close[i] > camarilla_s3_aligned[i]) or (regime == 'bull') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0