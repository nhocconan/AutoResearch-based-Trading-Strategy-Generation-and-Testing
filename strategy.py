#!/usr/bin/env python3
"""
4h_LongOnly_SR_Bounce_1dATR
Hypothesis: Go long near 1d support/resistance with 1d ATR stop. In trending markets, price often pulls back to SR levels before continuing. Enter long when price touches 1d SR ± 0.5*ATR and closes back inside, with 1d ADX > 25 (trend). Exit when price closes outside SR ± ATR or 1d ADX < 20 (range). Use volume > 1.5x 24-period average for confirmation. Designed for fewer trades (~20-30/year) by requiring trend + SR bounce + volume. Works in bull (buy dips) and bear (sell rallies) by only taking longs in uptrends (ADX > 25 and price > 200 EMA). In downtrends, remains flat. Uses 1d timeframe for SR/ADX to avoid noise.
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
    
    # Get 1d data for SR, ADX, ATR, EMA200
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for SR bands
    atr_period = 14
    atr_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= atr_period:
        tr = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        tr = np.concatenate([[np.nan], tr])
        atr_1d[atr_period-1] = np.nanmean(tr[1:atr_period])
        for i in range(atr_period, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # 1d ADX(14) for trend strength
    adx_period = 14
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr_adx = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.abs(high_1d[1:] - close_1d[:-1]),
        np.abs(low_1d[1:] - close_1d[:-1])
    )
    tr_adx = np.concatenate([[np.nan], tr_adx])
    
    atr_adx = np.full_like(close_1d, np.nan)
    if len(close_1d) >= adx_period:
        atr_adx[adx_period-1] = np.nanmean(tr_adx[1:adx_period])
        for i in range(adx_period, len(close_1d)):
            atr_adx[i] = (atr_adx[i-1] * (adx_period-1) + tr_adx[i]) / adx_period
    
    plus_di = np.full_like(close_1d, np.nan)
    minus_di = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 2*adx_period:
        smooth_plus_dm = np.full_like(plus_dm, np.nan)
        smooth_minus_dm = np.full_like(minus_dm, np.nan)
        smooth_plus_dm[adx_period-1] = np.nansum(plus_dm[1:adx_period])
        smooth_minus_dm[adx_period-1] = np.nansum(minus_dm[1:adx_period])
        for i in range(adx_period, len(close_1d)):
            smooth_plus_dm[i] = smooth_plus_dm[i-1] - (smooth_plus_dm[i-1]/adx_period) + plus_dm[i]
            smooth_minus_dm[i] = smooth_minus_dm[i-1] - (smooth_minus_dm[i-1]/adx_period) + minus_dm[i]
        plus_di = 100 * smooth_plus_dm / atr_adx
        minus_di = 100 * smooth_minus_dm / atr_adx
        dx = np.full_like(close_1d, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_1d = np.full_like(close_1d, np.nan)
        adx_1d[2*adx_period-1] = np.nanmean(dx[adx_period:2*adx_period])
        for i in range(2*adx_period, len(close_1d)):
            adx_1d[i] = (adx_1d[i-1] * (adx_period-1) + dx[i]) / adx_period
    else:
        adx_1d = np.full_like(close_1d, np.nan)
    
    # 1d EMA200 for trend filter
    ema200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema200_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema200_1d[i] = close_1d[i] * 2/201 + ema200_1d[i-1] * (1 - 2/201)
    
    # Support and Resistance: recent swing low/high (20-period)
    sr_period = 20
    support_1d = np.full_like(close_1d, np.nan)
    resistance_1d = np.full_like(close_1d, np.nan)
    for i in range(sr_period, len(close_1d)):
        support_1d[i] = np.min(low_1d[i-sr_period:i])
        resistance_1d[i] = np.max(high_1d[i-sr_period:i])
    
    # Align all 1d indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    support_1d_aligned = align_htf_to_ltf(prices, df_1d, support_1d)
    resistance_1d_aligned = align_htf_to_ltf(prices, df_1d, resistance_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = max(100, vol_period, 2*adx_period, sr_period)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(support_1d_aligned[i]) or 
            np.isnan(resistance_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 and price > EMA200 (uptrend only)
        is_uptrend = adx_1d_aligned[i] > 25 and close[i] > ema200_1d_aligned[i]
        
        # SR bands with ATR
        upper_band = resistance_1d_aligned[i] + atr_1d_aligned[i]
        lower_band = support_1d_aligned[i] - atr_1d_aligned[i]
        entry_zone_upper = resistance_1d_aligned[i] + 0.5 * atr_1d_aligned[i]
        entry_zone_lower = support_1d_aligned[i] - 0.5 * atr_1d_aligned[i]
        
        if position == 0:
            # Long: price touches SR zone and closes back inside, with trend and volume
            touched_support = low[i] <= entry_zone_lower and close[i] > entry_zone_lower
            touched_resistance = high[i] >= entry_zone_upper and close[i] < entry_zone_upper
            if (touched_support or touched_resistance) and is_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long exit: price closes outside SR ± ATR or trend weakens (ADX < 20)
            if close[i] > upper_band or close[i] < lower_band or adx_1d_aligned[i] < 20:
                signals[i] = 0.0  # exit long
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "4h_LongOnly_SR_Bounce_1dATR"
timeframe = "4h"
leverage = 1.0