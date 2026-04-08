#!/usr/bin/env python3
# 4h_donchian_breakout_volume_regime
# Hypothesis: Donchian channel breakout with volume confirmation and 1d regime filter.
# Long when price breaks above 4h Donchian(20) upper + volume > 1.5x 20-period avg + 1d ADX < 30 (range).
# Short when price breaks below 4h Donchian(20) lower + volume > 1.5x 20-period avg + 1d ADX < 30 (range).
# Exit when price returns to Donchian middle or ADX rises above 40 (trending).
# Uses discrete position sizing (0.25) to limit turnover.
# Target: 20-50 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian Channel (20)
    dc_period = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    for i in range(dc_period-1, n):
        dc_upper[i] = np.max(high[i-dc_period+1:i+1])
        dc_lower[i] = np.min(low[i-dc_period+1:i+1])
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    vol_surge = volume > 1.5 * vol_ma
    
    # 1d ADX regime filter (trending vs ranging)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM (Wilder smoothing = EMA with alpha=1/period)
    adx_period = 14
    alpha = 1.0 / adx_period
    
    atr = np.full(len(tr), np.nan)
    plus_di = np.full(len(tr), np.nan)
    minus_di = np.full(len(tr), np.nan)
    
    # Initialize first values
    if not np.isnan(tr[adx_period]):
        atr[adx_period] = np.nanmean(tr[1:adx_period+1])
        plus_dm_avg = np.nanmean(plus_dm[1:adx_period+1])
        minus_dm_avg = np.nanmean(minus_dm[1:adx_period+1])
        if atr[adx_period] > 0:
            plus_di[adx_period] = 100 * plus_dm_avg / atr[adx_period]
            minus_di[adx_period] = 100 * minus_dm_avg / atr[adx_period]
    
    # Wilder smoothing
    for i in range(adx_period+1, len(tr)):
        atr[i] = (atr[i-1] * (adx_period-1) + tr[i]) / adx_period
        plus_dm_avg = (plus_dm_avg * (adx_period-1) + plus_dm[i]) / adx_period
        minus_dm_avg = (minus_dm_avg * (adx_period-1) + minus_dm[i]) / adx_period
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_avg / atr[i]
            minus_di[i] = 100 * minus_dm_avg / atr[i]
    
    # Calculate DX and ADX
    dx = np.full(len(tr), np.nan)
    adx = np.full(len(tr), np.nan)
    
    for i in range(adx_period*2, len(tr)):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is EMA of DX
    for i in range(adx_period*2+1, len(tr)):
        if not np.isnan(dx[i]):
            if np.isnan(adx[i-1]):
                adx[i] = np.nanmean(dx[adx_period*2:i+1])
            else:
                adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align ADX to 4h timeframe (range condition: ADX < 30)
    adx_1d = adx
    adx_4h_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Regime: ranging market (ADX < 30) for mean reversion breakouts
    ranging = adx_4h_aligned < 30
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(dc_period, vol_ma_period, adx_period*2) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(dc_middle[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_4h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian middle OR ADX > 40 (trending)
            if close[i] < dc_middle[i] or adx_4h_aligned[i] > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian middle OR ADX > 40 (trending)
            if close[i] > dc_middle[i] or adx_4h_aligned[i] > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in ranging markets (ADX < 30)
            if ranging[i]:
                # Long entry: Price above upper Donchian with volume surge
                if close[i] > dc_upper[i] and vol_surge[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Price below lower Donchian with volume surge
                elif close[i] < dc_lower[i] and vol_surge[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals