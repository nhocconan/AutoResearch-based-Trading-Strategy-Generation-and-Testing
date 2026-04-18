#!/usr/bin/env python3
"""
1d_1w_TurtleTrend_v1
Hypothesis: Use 1w Donchian channels (55-period) for trend direction and 1d ADX for trend strength, with 1d volume confirmation. 
Go long when price breaks above 1w Donchian upper band AND ADX > 25 AND volume > 1.5x 20-period average. 
Go short when price breaks below 1w Donchian lower band AND ADX > 25 AND volume > 1.5x 20-period average.
Exit when price crosses the opposite Donchian band or ADX falls below 20.
Designed to capture major trends while avoiding whipsaws in ranging markets. Target: 15-25 trades/year.
Works in bull markets via trend following and in bear via short signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w Donchian channels (55-period)
    donch_len = 55
    upper_1w = np.full_like(high_1w, np.nan)
    lower_1w = np.full_like(low_1w, np.nan)
    
    if len(high_1w) >= donch_len:
        for i in range(donch_len, len(high_1w)):
            upper_1w[i] = np.max(high_1w[i-donch_len:i])
            lower_1w[i] = np.min(low_1w[i-donch_len:i])
    
    # Align Donchian channels to 1d timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # 1d ADX(14) for trend strength
    period = 14
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(low))
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(high, 1)), abs(low - np.roll(low, 1))))
    tr[0] = high[0] - low[0]
    
    atr = np.zeros(len(tr))
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.full(len(high), np.nan)
    minus_di = np.full(len(high), np.nan)
    dx = np.full(len(high), np.nan)
    
    if len(plus_dm) >= period:
        plus_sm = np.sum(plus_dm[:period])
        minus_sm = np.sum(minus_dm[:period])
        tr_sm = np.sum(tr[:period])
        
        for i in range(period, len(high)):
            plus_sm = plus_sm - (plus_sm / period) + plus_dm[i]
            minus_sm = minus_sm - (minus_sm / period) + minus_dm[i]
            tr_sm = tr_sm - (tr_sm / period) + tr[i]
            plus_di[i] = 100 * plus_sm / tr_sm if tr_sm != 0 else 0
            minus_di[i] = 100 * minus_sm / tr_sm if tr_sm != 0 else 0
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
    
    adx = np.full(len(high), np.nan)
    if len(dx) >= period:
        adx[period] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 1d timeframe (already in 1d, but ensuring alignment)
    adx_aligned = align_htf_to_ltf(prices, prices, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, period*2, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper + ADX > 25 + volume
            if close[i] > upper_1w_aligned[i] and adx_aligned[i] > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower + ADX > 25 + volume
            elif close[i] < lower_1w_aligned[i] and adx_aligned[i] > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 1w Donchian lower OR ADX < 20
            if close[i] < lower_1w_aligned[i] or adx_aligned[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 1w Donchian upper OR ADX < 20
            if close[i] > upper_1w_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_TurtleTrend_v1"
timeframe = "1d"
leverage = 1.0