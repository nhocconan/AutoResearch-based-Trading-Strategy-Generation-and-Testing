#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla pivot breakouts on daily timeframe with weekly trend filter and volume confirmation.
Uses 1d Camarilla levels (H3, L3) for breakout entries, 1w EMA50 for trend filter, and 1d volume spike (>1.5x 20-day average) for confirmation.
Designed for low trade frequency (~10-20/year) to work in both bull and bear markets via trend alignment and volume filter.
Breakouts at H3/L3 represent institutional order flow zones with higher reliability than standard support/resistance.
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
    
    # Get 1d data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-day average volume on 1d for volume confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels for 1d (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # We use H3 and L3 as breakout levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First value has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    rang = prev_high - prev_low
    H3 = prev_close + 1.1 * rang
    L3 = prev_close - 1.1 * rang
    
    # Align HTF 1w EMA50 to 1d timeframe (standard 1-bar delay for EMA)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Align HTF 1d Camarilla levels to 1d timeframe (no additional delay needed as they're based on prev day)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3, additional_delay_bars=0)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3, additional_delay_bars=0)
    
    # Align 1d volume MA to 1d timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for Camarilla breakout signals with trend filter and volume confirmation
            # Long: price breaks above H3 in uptrend (close > EMA50) with volume confirmation
            # Short: price breaks below L3 in downtrend (close < EMA50) with volume confirmation
            long_signal = (close[i] > H3_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and volume_confirm
            short_signal = (close[i] < L3_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below EMA50 (trend reversal) or breaks below L3 (mean reversion)
            exit_signal = (close[i] < ema50_1w_aligned[i]) or (close[i] < L3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal) or breaks above H3 (mean reversion)
            exit_signal = (close[i] > ema50_1w_aligned[i]) or (close[i] > H3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0