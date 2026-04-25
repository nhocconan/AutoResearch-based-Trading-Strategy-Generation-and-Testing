#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels act as strong support/resistance. Breakouts with volume confirmation,
aligned 1d EMA34 trend, and choppiness regime filter (CHOP < 61.8 = trending) capture strong momentum moves
while avoiding false breakouts in ranging markets. Designed for BTC/ETH with 75-200 total trades over 4 years.
Works in bull markets (trend continuation up) and bear markets (trend continuation down) by using 1d EMA34 as trend filter.
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
    
    # Get daily data for Camarilla pivot calculation and trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate for each day using previous day's OHLC (to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)  # previous day's high
    prev_low_1d = np.roll(low_1d, 1)    # previous day's low
    prev_close_1d = np.roll(close_1d, 1) # previous day's close
    
    # First day has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels for each day
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    camarilla_h3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Get 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Choppiness Index (CHOP) on 4h for regime filter
    # CHOP = 100 * log10(sum(ATR over period) / log10(highest_high - lowest_low) / log10(period))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    chop_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # For CHOP, we need highest high and lowest low over chop_period
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(chop_period, n):
        highest_high[i] = np.max(high[i-chop_period+1:i+1])
        lowest_low[i] = np.min(low[i-chop_period+1:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(chop_period, n):
        if atr[i] > 0 and highest_high[i] > lowest_low[i]:
            sum_atr = np.sum(tr[i-chop_period+1:i+1])
            chop[i] = 100 * np.log10(sum_atr) / np.log10(chop_period) / np.log10((highest_high[i] - lowest_low[i]) / atr[i])
        else:
            chop[i] = np.nan
    
    # Market is trending when CHOP < 61.8 (lower values = more trending)
    chop_filter = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and CHOP
    start_idx = max(34, 20, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        is_trending = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above H3 with volume confirmation in uptrend AND trending regime
            long_breakout = (curr_close > camarilla_h3) and volume_confirm and uptrend and is_trending
            # Short: price breaks below L3 with volume confirmation in downtrend AND trending regime
            short_breakout = (curr_close < camarilla_l3) and volume_confirm and downtrend and is_trending
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below L3 OR EMA34 trend turns down OR chop becomes too high (ranging)
            if curr_close < camarilla_l3 or curr_close < ema_34_val or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above H3 OR EMA34 trend turns up OR chop becomes too high (ranging)
            if curr_close > camarilla_h3 or curr_close > ema_34_val or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0