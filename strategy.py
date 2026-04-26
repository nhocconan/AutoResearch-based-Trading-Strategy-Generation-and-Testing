#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter, volume spike confirmation, and choppiness regime filter.
Long when price breaks above R1 with volume > 2.0x median volume, close > 12h EMA50, and CHOP > 50 (trending regime).
Short when price breaks below S1 with volume > 2.0x median volume, close < 12h EMA50, and CHOP > 50.
Uses discrete sizing (0.25) to minimize fee drag. Target: 75-150 trades over 4 years.
Works in bull/bear via 12h trend filter and regime filter to avoid false breakouts in ranging markets.
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
    
    # Calculate Camarilla levels for 4h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: volume > 2.0x 20-period median (more robust than mean)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 2.0)
    
    # Load 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 50 indicates trending regime (good for breakouts), CHOP < 50 indicates ranging
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Sum of true range over atr_period
    tr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    # Highest high and lowest low over atr_period
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(atr_period)
    # Avoid division by zero
    hl_range = hh - ll
    chop = np.zeros(n)
    for i in range(n):
        if hl_range[i] > 0 and not np.isnan(tr_sum[i]) and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / hl_range[i]) / np.log10(atr_period)
        else:
            chop[i] = 50.0  # neutral when undefined
    
    # Regime filter: CHOP > 50 indicates trending regime (favor breakouts)
    regime_filter = chop > 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period for EMA and 20 for volume median, 14 for CHOP)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 with volume confirmation, 12h uptrend, and trending regime
        long_condition = (close[i] > r1[i]) and volume_confirm[i] and (close[i] > ema_50_12h_aligned[i]) and regime_filter[i]
        # Short logic: break below S1 with volume confirmation, 12h downtrend, and trending regime
        short_condition = (close[i] < s1[i]) and volume_confirm[i] and (close[i] < ema_50_12h_aligned[i]) and regime_filter[i]
        
        # Exit logic: opposite Camarilla level touch or trend reversal or regime change to ranging
        exit_long = (close[i] < s1[i]) or (close[i] < ema_50_12h_aligned[i]) or (not regime_filter[i])
        exit_short = (close[i] > r1[i]) or (close[i] > ema_50_12h_aligned[i]) or (not regime_filter[i])
        
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

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0