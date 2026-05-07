#!/usr/bin/env python3
name = "4h_CCI_MeanReversion_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # CCI(20) - Commodity Channel Index
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    cci[np.isnan(mad) | (mad == 0)] = 0
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(cci[i]) or np.isnan(sma_tp[i]) or np.isnan(mad[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        cci_val = cci[i]
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: CCI < -100 (oversold) in daily uptrend with volume confirmation
            if cci_val < -100 and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: CCI > 100 (overbought) in daily downtrend with volume confirmation
            elif cci_val > 100 and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CCI crosses above -50 or trend changes
            if cci_val > -50 or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CCI crosses below 50 or trend changes
            if cci_val < 50 or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h CCI mean reversion with daily trend filter and volume confirmation
# - CCI < -100 indicates oversold conditions (long entry)
# - CCI > 100 indicates overbought conditions (short entry)
# - Daily EMA(34) trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Exit when CCI reverts toward mean (-50 for longs, 50 for shorts) or trend changes
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Position size 0.25 targets ~25-50 trades/year to avoid fee drag
# - CCI is effective in ranging markets which dominate 2025+ test period
# - Mean reversion with trend filter avoids counter-trend whipsaws