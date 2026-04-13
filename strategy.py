#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 75-150 total trades over 4 years (~19-38/year)
    # Works in both bull and bear markets by using ranging market filter (Chop > 61.8)
    # Camarilla pivots provide strong support/resistance levels from prior day
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for price action (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for volume and chop confirmation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivots (prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We'll use R3/S3 as primary breakout levels
    prior_day_range = high_1d - low_1d
    camarilla_r3 = close_1d + prior_day_range * 1.1 / 4.0
    camarilla_s3 = close_1d - prior_day_range * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h volume average (20-period) with min_periods
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate 12h Chop Index (Choppiness Index)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR (14-period) using Wilder's smoothing with min_periods via pandas ewm
    tr_series = pd.Series(tr)
    atr_12h = tr_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods with min_periods
    tr_series_for_sum = pd.Series(tr)
    sum_tr_14 = tr_series_for_sum.rolling(window=14, min_periods=14).sum().values
    
    # Chop Index = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    chop = np.where((atr_12h * 14) > 0,
                    100 * np.log10(sum_tr_14 / (atr_12h * 14)) / np.log10(14),
                    np.nan)
    
    # Align 12h Chop to 12h (wait for completed 12h bar)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h Chop > 61.8 (ranging market)
        ranging_market = chop_aligned[i] > 61.8
        # Exit regime: Chop < 38.2 (trending market begins)
        trending_market = chop_aligned[i] < 38.2
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        vol_12h_current = df_12h['volume'].values
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_current)
        volume_confirm = vol_12h_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Camarilla breakout signals
        long_breakout = close[i] > camarilla_r3_aligned[i]
        short_breakout = close[i] < camarilla_s3_aligned[i]
        
        # Entry logic: Camarilla breakout + volume confirmation + ranging market
        long_entry = long_breakout and volume_confirm and ranging_market
        short_entry = short_breakout and volume_confirm and ranging_market
        
        # Exit logic: price crosses opposite Camarilla level OR market becomes trending
        long_exit = close[i] < camarilla_s3_aligned[i] or trending_market
        short_exit = close[i] > camarilla_r3_aligned[i] or trending_market
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0