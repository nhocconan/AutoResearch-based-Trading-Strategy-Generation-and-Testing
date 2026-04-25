#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h timeframe with 1d EMA34 trend filter and volume confirmation (>2x average). Only trade when 1d ADX > 25 to ensure strong trending conditions. Uses discrete position sizing (0.30) to limit fee drawdown. Designed for low trade frequency (20-50/year) to survive bear markets like 2022 and range-bound periods like 2025.
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
    
    # Get 1d data for HTF trend and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        dm_plus_sum = pd.Series(dm_plus).rolling(window=window, min_periods=window).sum().values
        dm_minus_sum = pd.Series(dm_minus).rolling(window=window, min_periods=window).sum().values
        
        # Directional Indicators
        tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
        di_plus = 100 * dm_plus_sum / tr_sum_safe
        di_minus = 100 * dm_minus_sum / tr_sum_safe
        
        # DX and ADX
        dx = np.zeros(len(close))
        dx_denom = di_plus + di_minus
        dx_denom_safe = np.where(dx_denom == 0, 1e-10, dx_denom)
        dx = 100 * np.abs(di_plus - di_minus) / dx_denom_safe
        
        adx = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, window=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Camarilla levels (based on previous bar's OHLC)
    def calculate_camarilla(high, low, close):
        # Camarilla levels use previous period's OHLC
        # R4 = close + ((high-low) * 1.1/2)
        # R3 = close + ((high-low) * 1.1/4)
        # S3 = close - ((high-low) * 1.1/4)
        # S4 = close - ((high-low) * 1.1/2)
        # We use R1/S1 as breakout levels (closer to price for more sensitivity)
        range_hl = high - low
        r1 = close + (range_hl * 1.1 / 12)
        s1 = close - (range_hl * 1.1 / 12)
        return r1, s1
    
    # Shift by 1 to use previous bar's OHLC (no look-ahead)
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    camarilla_r1, camarilla_s1 = calculate_camarilla(high_shift, low_shift, close_shift)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(50, 20)  # EMA34 needs 34, Camarilla needs 20 (due to shift)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        df_1d_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(df_1d_close_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        htf_1d_bullish = df_1d_close_aligned[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = df_1d_close_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: strong spike (vol_ratio > 2.0)
        volume_confirmed = vol_ratio[i] > 2.0
        
        # Trend strength filter: only trade when ADX > 25 (strong trend)
        trend_filter = adx_1d_aligned[i] > 25.0
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation + strong trend
            long_setup = (close[i] > camarilla_r1[i]) and htf_1d_bullish and volume_confirmed and trend_filter
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation + strong trend
            short_setup = (close[i] < camarilla_s1[i]) and htf_1d_bearish and volume_confirmed and trend_filter
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches Camarilla S1 (opposite level) OR 1d trend turns bearish
            if (close[i] <= camarilla_s1[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches Camarilla R1 (opposite level) OR 1d trend turns bullish
            if (close[i] >= camarilla_r1[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0