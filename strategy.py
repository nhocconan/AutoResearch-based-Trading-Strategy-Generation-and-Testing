#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1d trend filter and volume confirmation.
Only trade breakouts in direction of 1d EMA34 trend: long when price > EMA34, short when price < EMA34.
Use volume spike (volume > 1.5 * ATR) to confirm breakout strength.
Target: 12-30 trades/year to minimize fee drag while capturing sustained moves.
Discrete sizing: 0.25.
Works in both bull and bear markets by filtering trades to trend direction only.
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
    
    # Get 1d data for trend regime and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volume spike filter (using 12h data)
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Calculate 12h Camarilla levels using previous 12h bar's OHLC
        # Need to access previous completed 12h bar
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Camarilla R1, S1, R3, S3 calculations
            range_val = prev_high - prev_low
            camarilla_r1 = prev_close + (range_val * 1.1 / 12)
            camarilla_s1 = prev_close - (range_val * 1.1 / 12)
            camarilla_r3 = prev_close + (range_val * 1.1 / 4)
            camarilla_s3 = prev_close - (range_val * 1.1 / 4)
        else:
            camarilla_r1 = camarilla_s1 = camarilla_r3 = camarilla_s3 = close[i]
        
        # Volume spike: current volume > 1.5 * ATR (adaptive threshold)
        volume_spike = volume[i] > 1.5 * atr[i]
        
        # Determine 1d trend regime
        # Bull regime: price > EMA34
        # Bear regime: price < EMA34
        if close[i] > ema_34_1d_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1d_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades (should rarely happen)
        
        if position == 0:
            # Long setup: price breaks above R1 with volume spike AND bull regime
            long_setup = (close[i] > camarilla_r1) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below S1 with volume spike AND bear regime
            short_setup = (close[i] < camarilla_s1) and volume_spike and (regime == 'bear')
            
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
            # Exit: price closes below S3 (reversal) OR regime turns bearish OR max hold (12 bars = 6 days)
            if (close[i] < camarilla_s3) or (regime == 'bear') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price closes above R3 (reversal) OR regime turns bullish OR max hold (12 bars = 6 days)
            if (close[i] > camarilla_r3) or (regime == 'bull') or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0