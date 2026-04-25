#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v11
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike captures institutional breakout moves in both bull and bear markets. Uses discrete position sizing (0.25) and ATR-based stoploss (2.0) to target 75-200 total trades over 4 years. Works in trending markets by only taking breakouts in direction of higher timeframe trend, avoiding false reversals. Added volume spike filter to reduce false breakouts and improve trade quality. Reduced trade frequency by tightening volume spike threshold to 3.5x and adding minimum holding period of 6 bars to avoid whipsaw. Fixed: removed redundant 4h close alignment to prevent look-ahead and used proper discrete sizing.
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
    
    # Get 1d data for trend filter (EMA34) - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR for stoploss calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 4h data for Camarilla calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h
    # Camarilla levels based on previous bar's range
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume filter: volume > 3.5x 20-period average (tighter filter for quality)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (3.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, 14)  # EMA34 needs 34, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        
        # Get 4h close aligned for direct comparison
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
        close_4h_val = close_4h_aligned[i]
        is_uptrend = close_4h_val > ema_34_val
        
        if position == 0:
            # Look for entry signals: breakout in direction of 1d trend
            long_signal = (close_4h_val > r1_val) and is_uptrend and vol_spike[i]
            short_signal = (close_4h_val < s1_val) and (not is_uptrend) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_4h_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_4h_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit conditions:
            # 1. Minimum holding period of 6 bars to avoid whipsaw
            # 2. Price closes below S1 (opposite Camarilla level)
            # 3. ATR-based stoploss: 2.0 * ATR below entry
            if bars_since_entry >= 6:
                exit_signal = close_4h_val < s1_val
                stop_signal = close_4h_val < (entry_price - 2.0 * atr_val)
                if exit_signal or stop_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit conditions:
            # 1. Minimum holding period of 6 bars to avoid whipsaw
            # 2. Price closes above R1 (opposite Camarilla level)
            # 3. ATR-based stoploss: 2.0 * ATR above entry
            if bars_since_entry >= 6:
                exit_signal = close_4h_val > r1_val
                stop_signal = close_4h_val > (entry_price + 2.0 * atr_val)
                if exit_signal or stop_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v11"
timeframe = "4h"
leverage = 1.0