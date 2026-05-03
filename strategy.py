#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with 1d volume spike and choppiness regime filter
# TRIX (15-period) identifies momentum shifts. Long when TRIX crosses above zero in 1d uptrend with volume spike and choppy market (CHOP > 61.8).
# Short when TRIX crosses below zero in 1d downtrend with volume spike and choppy market.
# Volume spike confirms conviction. Chop filter avoids whipsaws in strong trends.
# Designed for 20-40 trades/year on 4h to minimize fee drag. Works in both bull and bear markets by capturing momentum in ranging conditions.

name = "4h_TRIX_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (14-period)
    atr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                 abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                 abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
        atr_1d[i] = tr if i == 1 else (atr_1d[i-1] * 13 + tr) / 14
    atr_1d[0] = atr_1d[1] if len(df_1d) > 1 else 0
    
    sum_tr_14 = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        sum_tr = 0
        for j in range(i-13, i+1):
            tr = max(df_1d['high'].iloc[j] - df_1d['low'].iloc[j],
                     abs(df_1d['high'].iloc[j] - df_1d['close'].iloc[j-1] if j > 0 else df_1d['open'].iloc[j]),
                     abs(df_1d['low'].iloc[j] - df_1d['close'].iloc[j-1] if j > 0 else df_1d['open'].iloc[j]))
            sum_tr += tr
        sum_tr_14[i] = sum_tr
    
    chop_1d = np.full(len(df_1d), 50.0)
    for i in range(14, len(df_1d)):
        if sum_tr_14[i] > 0 and atr_1d[i] > 0:
            chop_1d[i] = 100 * np.log10(sum_tr_14[i] / (atr_1d[i] * 14)) / np.log10(14)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate TRIX (15-period) on 4h close
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.zeros(n)
    trix[14:] = (ema3[15:] - ema3[14:-1]) / ema3[14:-1] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(15, n):  # Start after sufficient warmup for TRIX
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(trix[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Choppiness regime: choppy market (CHOP > 61.8)
        choppy_market = chop_1d_aligned[i] > 61.8
        
        # TRIX signals
        trix_cross_above = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_below = trix[i] < 0 and trix[i-1] >= 0
        
        if position == 0:
            # Long: TRIX crosses above zero in 1d uptrend with volume spike and choppy market
            if trix_cross_above and ema_34_1d_aligned[i] > close[i] and volume_spike and choppy_market:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero in 1d downtrend with volume spike and choppy market
            elif trix_cross_below and ema_34_1d_aligned[i] < close[i] and volume_spike and choppy_market:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or loses 1d uptrend
            if trix_cross_below or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or loses 1d downtrend
            if trix_cross_above or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals