#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v10
Hypothesis: Tighten entry further by requiring volume spike >10.0x average and adding ADX trend strength filter (ADX > 25) to avoid false breakouts. Target 10-20 trades/year to minimize fee drag while maintaining edge in both bull and bear markets via Camarilla pivot breaks with 1d trend alignment.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (proven stability)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 4h volume ratio (current vs 20-period average = 10h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # Calculate 4h ADX for trend strength filter
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Directional Movement
        dm_plus = np.zeros(len(close))
        dm_minus = np.zeros(len(close))
        for i in range(1, len(close)):
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            dm_plus[i] = up if up > down and up > 0 else 0
            dm_minus[i] = down if down > up and down > 0 else 0
        
        # Smoothed values
        atr = np.zeros(len(close))
        atr_smooth = np.zeros(len(close))
        dm_plus_smooth = np.zeros(len(close))
        dm_minus_smooth = np.zeros(len(close))
        
        # Initial values
        if len(close) >= window:
            atr[window-1] = np.mean(tr[1:window])
            dm_plus_smooth[window-1] = np.mean(dm_plus[1:window])
            dm_minus_smooth[window-1] = np.mean(dm_minus[1:window])
            
            # Wilder's smoothing
            for i in range(window, len(close)):
                atr_smooth[i] = (atr_smooth[i-1] * (window-1) + tr[i]) / window
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (window-1) + dm_plus[i]) / window
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (window-1) + dm_minus[i]) / window
        
        # Directional Indicators
        di_plus = np.zeros(len(close))
        di_minus = np.zeros(len(close))
        dx = np.zeros(len(close))
        for i in range(window, len(close)):
            if atr_smooth[i] > 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr_smooth[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr_smooth[i]
                if di_plus[i] + di_minus[i] > 0:
                    dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX (smoothed DX)
        adx = np.zeros(len(close))
        if len(close) >= 2*window-1:
            adx[2*window-2] = np.mean(dx[window:2*window-1])
            for i in range(2*window-1, len(close)):
                adx[i] = (adx[i-1] * (window-1) + dx[i]) / window
        
        return adx
    
    adx = calculate_adx(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA, volume MA, and ADX
    start_idx = max(34, 20, 14*2)  # EMA34 needs 34, vol MA needs 20, ADX needs ~28
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Determine 1d trend (bullish = price above EMA34)
        df_1d_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(df_1d_close_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        htf_1d_bullish = df_1d_close_aligned[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = df_1d_close_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: need significant spike (vol_ratio > 10.0) - much stricter
        volume_confirmed = vol_ratio[i] > 10.0
        
        # ADX trend strength filter: only trade when trend is strong (ADX > 25)
        trend_filter = adx[i] > 25.0 if not np.isnan(adx[i]) else False
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation + trend strength
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_1d_bullish and volume_confirmed and trend_filter
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation + trend strength
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_1d_bearish and volume_confirmed and trend_filter
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 (opposite level) OR 1d trend turns bearish
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (opposite level) OR 1d trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v10"
timeframe = "4h"
leverage = 1.0