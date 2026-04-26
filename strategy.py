#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime
Hypothesis: 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume confirmation (>2.0x average), and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trending). 
In trending regimes (CHOP < 38.2), trade breakouts in direction of 1d EMA34 trend. 
In ranging regimes (CHOP > 61.8), trade mean reversion at Camarilla H3/L3 levels. 
ATR(14) trailing stop (2.5x) manages risk. Discrete sizing 0.30 targets ~25 trades/year to minimize fee drag.
Designed for both bull and bear markets: regime filter adapts to market conditions, ensuring trades only occur in favorable environments.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) on 4h for breakout confirmation and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) on 1d for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    atr_1d = []
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.concatenate([[np.nan], atr_1d])
    atr_1d_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    high_max = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_min = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log(high_max - low_min) / np.log(14)
    chop = 100 * np.log10(atr_1d_sum / chop_denom) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Camarilla levels from prior 1d bar
    if len(high_1d) < 2:
        camarilla_h3 = np.full_like(close_1d_arr, np.nan)
        camarilla_l3 = np.full_like(close_1d_arr, np.nan)
        camarilla_h4 = np.full_like(close_1d_arr, np.nan)
        camarilla_l4 = np.full_like(close_1d_arr, np.nan)
        camarilla_r1 = np.full_like(close_1d_arr, np.nan)
        camarilla_s1 = np.full_like(close_1d_arr, np.nan)
    else:
        high_1d_prev = high_1d[:-1]
        low_1d_prev = low_1d[:-1]
        close_1d_prev = close_1d[:-1]
        camarilla_h3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 4
        camarilla_l3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 4
        camarilla_h4 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 2
        camarilla_l4 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 2
        camarilla_r1 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 12
        camarilla_s1 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 12
        camarilla_h3 = np.concatenate([[np.nan], camarilla_h3])
        camarilla_l3 = np.concatenate([[np.nan], camarilla_l3])
        camarilla_h4 = np.concatenate([[np.nan], camarilla_h4])
        camarilla_l4 = np.concatenate([[np.nan], camarilla_l4])
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of volume MA (20), 1d EMA (34), ATR (14), CHOP (14)
    start_idx = max(20, 34, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        h3_val = camarilla_h3_aligned[i]
        l3_val = camarilla_l3_aligned[i]
        h4_val = camarilla_h4_aligned[i]
        l4_val = camarilla_l4_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        chop_val = chop_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict for quality)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        # Breakout threshold: price must close beyond Camarilla level by 1.5*ATR (balanced for sensitivity)
        breakout_threshold = 1.5 * atr_val
        
        if position == 0:
            if chop_val < 38.2:  # Trending regime
                # Long: close above R1 + threshold, uptrend (close > EMA34_1d), volume confirmation
                long_signal = (close_val > r1_val + breakout_threshold) and (close_val > ema_34_1d_val) and volume_confirmed
                # Short: close below S1 - threshold, downtrend (close < EMA34_1d), volume confirmation
                short_signal = (close_val < s1_val - breakout_threshold) and (close_val < ema_34_1d_val) and volume_confirmed
                
                if long_signal:
                    signals[i] = 0.30
                    position = 1
                    entry_price = close_val
                    highest_since_entry = close_val
                elif short_signal:
                    signals[i] = -0.30
                    position = -1
                    entry_price = close_val
                    lowest_since_entry = close_val
                else:
                    signals[i] = 0.0
            elif chop_val > 61.8:  # Ranging regime
                # Long: mean reversion at L3, price < L3 + 0.5*ATR, volume confirmation
                long_signal = (close_val < l3_val + 0.5 * atr_val) and volume_confirmed
                # Short: mean reversion at H3, price > H3 - 0.5*ATR, volume confirmation
                short_signal = (close_val > h3_val - 0.5 * atr_val) and volume_confirmed
                
                if long_signal:
                    signals[i] = 0.30
                    position = 1
                    entry_price = close_val
                    highest_since_entry = close_val
                elif short_signal:
                    signals[i] = -0.30
                    position = -1
                    entry_price = close_val
                    lowest_since_entry = close_val
                else:
                    signals[i] = 0.0
            else:  # Neutral regime, no trading
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: price closes below L3 in ranging regime or S1 in trending regime
            elif chop_val > 61.8 and close_val < l3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            elif chop_val < 38.2 and close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: trend reversal (close below EMA34_1d)
            elif close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: price closes above H3 in ranging regime or R1 in trending regime
            elif chop_val > 61.8 and close_val > h3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            elif chop_val < 38.2 and close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: trend reversal (close above EMA34_1d)
            elif close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0