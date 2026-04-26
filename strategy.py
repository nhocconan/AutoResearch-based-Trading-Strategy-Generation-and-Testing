#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopFilter
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter, volume spike (>2x average), and chop regime filter (CHOP > 61.8 = range, mean reversion). 
In range markets: price breaks above R1 with 1d uptrend and volume → long; breaks below S1 with 1d downtrend and volume → short. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 4h timeframe.
Requires BTC/ETH edge via 1d trend, volume, and regime filters; avoids SOL-only bias by requiring multi-factor confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Chopiness Index on 1d (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr_sum = np.zeros(len(close_arr))
        true_range = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = abs(high_arr[i] - close_arr[i-1])
            lc = abs(low_arr[i] - close_arr[i-1])
            true_range[i] = max(hl, hc, lc)
        
        # ATR calculation with smoothing
        atr_sum[period-1] = np.sum(true_range[1:period]) if period <= len(true_range) else 0
        for i in range(period, len(close_arr)):
            atr_sum[i] = (atr_sum[i-1] * (period-1) + true_range[i]) / period
        
        # Chop calculation: 100 * log10(ATR_sum / (max_high - min_low)) / log10(period)
        chop = np.full(len(close_arr), 50.0)  # default neutral
        for i in range(period, len(close_arr)):
            if atr_sum[i] > 0:
                max_high = np.max(high_arr[i-period+1:i+1])
                min_low = np.min(low_arr[i-period+1:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(atr_sum[i] * period / (max_high - min_low)) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1 for Camarilla calculation, 34 for EMA, 14 for chop, 20 for volume)
    start_idx = max(1, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous day's OHLC
        # For 4h timeframe, previous day = previous 6 bars
        prev_1d_idx = i - 6
        if prev_1d_idx < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        prev_high = high[prev_1d_idx]
        prev_low = low[prev_1d_idx]
        prev_close = close[prev_1d_idx]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R1 and S1 levels
        R1 = prev_close + (range_val * 1.1 / 12)
        S1 = prev_close - (range_val * 1.1 / 12)
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(R1) or np.isnan(S1) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(chop_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Chop regime filter: only trade in range markets (CHOP > 61.8)
        chop_filter = chop_val > 61.8
        
        # Long logic: price breaks above R1 with 1d uptrend, volume confirmation, and chop filter
        long_condition = (close_val > R1) and (close_val > ema_val) and volume_confirmed and chop_filter
        # Short logic: price breaks below S1 with 1d downtrend, volume confirmation, and chop filter
        short_condition = (close_val < S1) and (close_val < ema_val) and volume_confirmed and chop_filter
        
        # Exit logic: trend reversal or opposite breakout
        exit_long = (close_val < ema_val) or (close_val < S1)
        exit_short = (close_val > ema_val) or (close_val > R1)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0