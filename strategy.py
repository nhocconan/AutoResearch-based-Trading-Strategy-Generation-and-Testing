#!/usr/bin/env python3
# 1d_CCI_Trend_Reversal
# Hypothesis: Daily CCI combined with weekly trend filter and volume confirmation captures mean-reversion
# opportunities during pullbacks in strong trends. Works in bull markets (buy dips) and bear markets
# (sell rallies) by using 1-week trend as filter. Target: 10-25 trades/year with low churn.

name = "1d_CCI_Trend_Reversal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily data
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    # Avoid division by zero
    cci = np.where(mad > 0, (tp - ma_tp) / (0.015 * mad), 0.0)
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (20-period MA on daily chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for CCI, 34 for weekly EMA, 20 for volume MA
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(cci[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # CCI signals: oversold/overbought with trend alignment
        cci_oversold = cci[i] < -100
        cci_overbought = cci[i] > 100
        
        if position == 0:
            # Long entry: CCI oversold + weekly uptrend + volume confirmation
            if cci_oversold and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: CCI overbought + weekly downtrend + volume confirmation
            elif cci_overbought and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI crosses above zero or weekly trend turns down
            if cci[i] > 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI crosses below zero or weekly trend turns up
            if cci[i] < 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals