#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h strategy using 1d Camarilla R1/S1 breakouts filtered by 1d EMA34 trend and volume spike (>2x 20-period average). Uses chop regime filter (CHOP < 38.2 = trending) to avoid false breakouts in ranging markets. Enters long when price breaks above 1d R1 AND 1d close > EMA34 (uptrend) AND volume confirmation AND trending regime. Enters short when price breaks below 1d S1 AND 1d close < EMA34 (downtrend) AND volume confirmation AND trending regime. Exits on mean reversion to 1d close or trend break. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull/bear via 1d trend filter and volume confirmation.
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
    hl_range_1d = df_1d['high'].values - df_1d['low'].values
    camarilla_r1_1d = df_1d['close'].values + hl_range_1d * 1.1 / 12
    camarilla_s1_1d = df_1d['close'].values - hl_range_1d * 1.1 / 12
    
    # Align 1d indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter: only allow breakouts in trending markets (CHOP < 38.2 = trending)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[0], tr1])  # align length
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 * 14 / np.log10(14) / (max_high - min_low + 1e-10))
    chop_regime = chop < 38.2  # True = trending market (allow breakouts)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), chop (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_1d_aligned[i]) or np.isnan(camarilla_s1_1d_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r1_1d_val = camarilla_r1_1d_aligned[i]
        s1_1d_val = camarilla_s1_1d_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_regime[i]
        close_1d_val = close_1d_aligned[i]
        
        if position == 0:
            # Look for entry: price breakout above R1 (long) or below S1 (short) with trend, volume, and regime filter
            # Long: price > R1 AND 1d uptrend AND volume confirmation AND trending regime
            long_condition = (close_val > r1_1d_val) and (close_val > ema_1d_val) and vol_conf and chop_ok
            # Short: price < S1 AND 1d downtrend AND volume confirmation AND trending regime
            short_condition = (close_val < s1_1d_val) and (close_val < ema_1d_val) and vol_conf and chop_ok
            
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0