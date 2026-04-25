#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R1 in bull regime (price > 1d EMA34), short when breaks below S1 in bear regime (price < 1d EMA34).
Volume confirmation requires volume > 1.5 * ATR(14) to avoid false breakouts.
Only trade in direction of 1d trend: long in bull regime, short in bear regime, flat in range.
Discrete sizing: 0.25 to minimize fee drag while capturing sustained moves.
Target: 20-50 trades/year to stay within proven winning range.
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
    
    # Get 1d data for trend regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Calculate Camarilla levels for current 4h bar using previous 4h bar's OHLC
        if i >= 1:
            phigh = high[i-1]
            plow = low[i-1]
            pclose = close[i-1]
            range_val = phigh - plow
            
            # Camarilla levels
            R1 = pclose + range_val * 1.1 / 12
            S1 = pclose - range_val * 1.1 / 12
        else:
            R1 = close[i]
            S1 = close[i]
        
        # Volume spike: current volume > 1.5 * ATR
        volume_spike = volume[i] > 1.5 * atr[i]
        
        # Determine 1d trend regime
        # Bull regime: price > EMA34
        # Bear regime: price < EMA34
        # Range regime: near EMA34 (within 0.5*ATR of 1d equivalent)
        # Convert 1d ATR to 4h equivalent threshold: 1d = 6*4h bars, so ATR_1d ≈ ATR_4h * sqrt(6)
        atr_4h = atr[i]
        atr_1d_approx = atr_4h * np.sqrt(6)  # rough approximation
        regime_threshold = 0.5 * atr_1d_approx
        
        if close[i] > ema_34_1d_aligned[i] + regime_threshold:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1d_aligned[i] - regime_threshold:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades
        
        if position == 0:
            # Long setup: price breaks above R1 AND volume spike AND bull regime
            long_setup = (close[i] > R1) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below S1 AND volume spike AND bear regime
            short_setup = (close[i] < S1) and volume_spike and (regime == 'bear')
            
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
            # Exit: price closes below R1 OR regime turns bearish OR max holding period (24 bars = 4 days)
            if (close[i] < R1) or (regime == 'bear') or (bars_since_entry >= 24):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price closes above S1 OR regime turns bullish OR max holding period (24 bars = 4 days)
            if (close[i] > S1) or (regime == 'bull') or (bars_since_entry >= 24):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0