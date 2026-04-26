#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and chop regime filter (CHOP > 50) captures high-probability institutional breakouts with controlled trade frequency. The chop filter avoids whipsaws in ranging markets, improving performance in both bull and bear regimes. Discrete sizing (0.25) minimizes fee churn. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Choppiness Index on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    atr_period = 14
    chop_period = 14
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(sum_atr / (np.log10(chop_period) * (max_high - min_low)))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range * 1.1 / 12
    s1 = prev_close - 1.1 * camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, chop_period)  # EMA34, vol MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        chop_val = chop_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        # Chop regime filter: only trade when market is not too choppy (CHOP < 50)
        regime_ok = chop_val < 50.0
        
        if position == 0:
            # Look for entry signals: Camarilla R1/S1 breakout with trend, volume, and regime
            # Long: price breaks above R1 with uptrend (close > EMA34), volume spike, and regime ok
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and regime_ok
            # Short: price breaks below S1 with downtrend (close < EMA34), volume spike, and regime ok
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and regime_ok
            
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
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0