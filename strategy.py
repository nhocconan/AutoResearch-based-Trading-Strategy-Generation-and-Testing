#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume spike confirmation.
- Williams %R(14) from 6h: long when < -80 (oversold), short when > -20 (overbought)
- 1d ADX(14) > 25 confirms strong trend (avoid whipsaw in ranging markets)
- Volume spike: current 6h volume > 1.5 * 20-period 6h volume average
- Entry only in direction of 1d trend (ADX + DI+ > DI- for long, DI- > DI+ for short)
- Exit when Williams %R returns to -50 (mean reversion) or trend weakens (ADX < 20)
- Designed to catch extreme reversals in strong trends with institutional volume
- Signal size: 0.25 discrete levels
- Target: 75-175 total trades over 4 years (19-44/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d OHLC for ADX and DI (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX and DI
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.maximum(high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]))], tr1])
    
    tr2 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr2, np.abs(low_1d[1:] - close_1d[:-1]))
    tr2 = np.concatenate([[np.maximum(high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]))], tr2])
    
    tr3 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr3 = np.maximum(tr3, np.abs(low_1d[1:] - close_1d[:-1]))
    tr3 = np.concatenate([[np.maximum(high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]))], tr3])
    
    # Actually, simpler approach: use pandas for 1d calculations
    df_1d_copy = df_1d.copy()
    # Calculate +DM and -DM
    plus_dm = df_1d_copy['high'].diff()
    minus_dm = df_1d_copy['low'].diff().multiply(-1)
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    # Fix first row
    plus_dm.iloc[0] = 0
    minus_dm.iloc[0] = 0
    
    # True Range for 1d
    tr_temp = pd.DataFrame()
    tr_temp['h-l'] = df_1d_copy['high'] - df_1d_copy['low']
    tr_temp['h-pc'] = np.abs(df_1d_copy['high'] - df_1d_copy['close'].shift(1))
    tr_temp['l-pc'] = np.abs(df_1d_copy['low'] - df_1d_copy['close'].shift(1))
    tr_temp['tr'] = tr_temp.max(axis=1)
    tr_temp.iloc[0, tr_temp.columns.get_loc('tr')] = df_1d_copy['high'].iloc[0] - df_1d_copy['low'].iloc[0]
    
    # Wilder's smoothing
    period = 14
    atr_1d = tr_temp['tr'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di_1d = 100 * (plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_1d)
    minus_di_1d = 100 * (minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = dx_1d.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Values and alignment
    adx_1d_vals = adx_1d.values
    plus_di_1d_vals = plus_di_1d.values
    minus_di_1d_vals = minus_di_1d.values
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_vals)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d_vals)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d_vals)
    
    # 6h Williams %R(14)
    def calc_williams_r(high_arr, low_arr, close_arr, lookback):
        highest_high = pd.Series(high_arr).rolling(window=lookback, min_periods=lookback).max()
        lowest_low = pd.Series(low_arr).rolling(window=lookback, min_periods=lookback).min()
        wr = -100 * (highest_high - close_arr) / (highest_high - lowest_low)
        return wr.fillna(-50).values  # Neutral when not enough data
    
    wr_6h = calc_williams_r(high, low, close, 14)
    
    # 6h volume average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # Trend conditions
    uptrend_1d = (adx_1d_aligned > 25) & (plus_di_1d_aligned > minus_di_1d_aligned)
    downtrend_1d = (adx_1d_aligned > 25) & (minus_di_1d_aligned > plus_di_1d_aligned)
    weak_trend = adx_1d_aligned < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 30)  # Williams %R needs 14, volume MA needs 20, 1d ADX needs 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr_6h[i]) or np.isnan(vol_ma_6h[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(plus_di_1d_aligned[i]) or 
            np.isnan(minus_di_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend AND volume spike
            if wr_6h[i] < -80 and uptrend_1d[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume spike
            elif wr_6h[i] > -20 and downtrend_1d[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 OR trend weakens
            if wr_6h[i] >= -50 or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 OR trend weakens
            if wr_6h[i] <= -50 or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADXTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0