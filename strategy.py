#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend filter (EMA21) and 1d momentum filter (ROC25) for direction,
# with entry on 1h pullbacks to the 21 EMA during strong trends. Designed to work in both bull
# and bear markets by following the higher timeframe trend. Uses volume confirmation to avoid
# false breakouts. Target: 15-35 trades/year per symbol to minimize fee drag.
name = "1h_EMA21_Pullback_4hTrend_1dROC"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Load 1d data for momentum filter (ROC25)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # 4h EMA21 for trend direction
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d ROC25 for momentum strength
    roc_25_1d = (pd.Series(df_1d['close']).pct_change(25) * 100).values
    roc_25_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_25_1d)
    
    # 1h EMA21 for entry timing (pullback to EMA)
    ema_21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume above 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_above_ema = volume > vol_ema_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(roc_25_1d_aligned[i]) or 
            np.isnan(ema_21_1h[i]) or np.isnan(vol_above_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and momentum filters
        uptrend = ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1]  # 4h EMA rising
        strong_uptrend = uptrend and (roc_25_1d_aligned[i] > 0)  # 4h EMA up + 1d ROC positive
        downtrend = ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1]  # 4h EMA falling
        strong_downtrend = downtrend and (roc_25_1d_aligned[i] < 0)  # 4h EMA down + 1d ROC negative
        
        if position == 0:
            # Long entry: pullback to EMA in strong uptrend with volume
            near_ema = low[i] <= ema_21_1h[i] * 1.005  # within 0.5% of EMA (low touches or slightly above)
            long_condition = strong_uptrend and near_ema and vol_above_ema[i]
            
            # Short entry: pullback to EMA in strong downtrend with volume
            near_ema_short = high[i] >= ema_21_1h[i] * 0.995  # within 0.5% of EMA (high touches or slightly below)
            short_condition = strong_downtrend and near_ema_short and vol_above_ema[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price moves significantly above EMA
            if not strong_uptrend or high[i] > ema_21_1h[i] * 1.02:  # 2% above EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or price moves significantly below EMA
            if not strong_downtrend or low[i] < ema_21_1h[i] * 0.98:  # 2% below EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals