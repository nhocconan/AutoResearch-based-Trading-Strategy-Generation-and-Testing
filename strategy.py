#!/usr/bin/env python3
"""
1d_1W_Donchian_Breakout_VolumeTrend_v1
Hypothesis: Use weekly Donchian breakout for trend direction and daily volume confirmation.
Go long when price breaks above weekly Donchian upper band with volume > 2x daily average,
short when price breaks below weekly Donchian lower band with volume > 2x daily average.
Uses daily ADX filter to avoid choppy markets (ADX < 25). 
Target: 10-20 trades/year by requiring multiple confirmations to reduce noise.
Works in bull markets via trend following and in bear via short signals.
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
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly Donchian channels (20-period)
    donch_len = 20
    upper_weekly = np.full_like(high_weekly, np.nan)
    lower_weekly = np.full_like(low_weekly, np.nan)
    
    if len(high_weekly) >= donch_len:
        for i in range(donch_len, len(high_weekly)):
            upper_weekly[i] = np.max(high_weekly[i-donch_len:i])
            lower_weekly[i] = np.min(low_weekly[i-donch_len:i])
    
    # Align weekly Donchian channels to daily timeframe
    upper_weekly_aligned = align_htf_to_ltf(prices, df_weekly, upper_weekly)
    lower_weekly_aligned = align_htf_to_ltf(prices, df_weekly, lower_weekly)
    
    # Get daily data for ADX and volume
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Daily ADX(14) for trend strength
    adx_period = 14
    tr = np.maximum(high_daily[1:] - low_daily[1:], 
                    np.maximum(np.abs(high_daily[1:] - close_daily[:-1]), 
                               np.abs(low_daily[1:] - close_daily[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    plus_dm = np.where((high_daily[1:] - high_daily[:-1]) > (low_daily[:-1] - low_daily[1:]), 
                       np.maximum(high_daily[1:] - high_daily[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_daily[:-1] - low_daily[1:]) > (high_daily[1:] - high_daily[:-1]), 
                        np.maximum(low_daily[:-1] - low_daily[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    atr = np.full_like(tr, np.nan)
    if len(tr) >= adx_period:
        atr[adx_period] = np.nanmean(tr[1:adx_period+1])
        for i in range(adx_period+1, len(tr)):
            atr[i] = (atr[i-1] * (adx_period - 1) + tr[i]) / adx_period
    
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    if len(atr) >= adx_period and not np.all(np.isnan(atr[adx_period:])):
        plus_di_smoothed = np.full_like(tr, np.nan)
        minus_di_smoothed = np.full_like(tr, np.nan)
        
        plus_di_smoothed[adx_period] = np.nansum(plus_dm[1:adx_period+1])
        minus_di_smoothed[adx_period] = np.nansum(minus_dm[1:adx_period+1])
        
        for i in range(adx_period+1, len(tr)):
            plus_di_smoothed[i] = (plus_di_smoothed[i-1] * (adx_period - 1) + plus_dm[i]) / adx_period
            minus_di_smoothed[i] = (minus_di_smoothed[i-1] * (adx_period - 1) + minus_dm[i]) / adx_period
        
        plus_di = np.where(atr != 0, 100 * plus_di_smoothed / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_di_smoothed / atr, 0)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.full_like(tr, np.nan)
    if len(dx) >= adx_period:
        adx[2*adx_period-1] = np.nanmean(dx[adx_period:2*adx_period])
        for i in range(2*adx_period, len(tr)):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align daily ADX to daily timeframe (already aligned)
    adx_aligned = adx
    
    # Daily volume confirmation: volume > 2x 20-day average
    vol_ma = np.full_like(volume_daily, np.nan)
    vol_period = 20
    
    if len(volume_daily) >= vol_period:
        for i in range(vol_period, len(volume_daily)):
            vol_ma[i] = np.mean(volume_daily[i-vol_period:i])
    
    vol_ratio = np.where(vol_ma > 0, volume_daily / vol_ma, 0)
    
    # Align weekly Donchian and daily volume ratio to daily timeframe
    upper_weekly_aligned_d = align_htf_to_ltf(prices, df_weekly, upper_weekly)
    lower_weekly_aligned_d = align_htf_to_ltf(prices, df_weekly, lower_weekly)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_daily, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, adx_period*2, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_weekly_aligned_d[i]) or np.isnan(lower_weekly_aligned_d[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: volume > 2x average
        vol_confirm = vol_ratio_aligned[i] > 2.0
        
        if position == 0 and strong_trend:
            # Long: price breaks above weekly Donchian upper + volume confirmation
            if close[i] > upper_weekly_aligned_d[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower + volume confirmation
            elif close[i] < lower_weekly_aligned_d[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly Donchian lower OR ADX < 20 (trend weakening)
            if close[i] < lower_weekly_aligned_d[i] or adx_aligned[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly Donchian upper OR ADX < 20 (trend weakening)
            if close[i] > upper_weekly_aligned_d[i] or adx_aligned[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Donchian_Breakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0