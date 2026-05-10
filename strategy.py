#!/usr/bin/env python3
# 4H_CCI_MeanReversion_1dTrend_VolumeFilter
# Hypothesis: Uses CCI(20) for mean reversion on 4h chart, filtered by 1-day trend (close > EMA50) and volume spike (>1.5x 20-period average).
# Enters long when CCI crosses below -100 in uptrend with volume confirmation.
# Enters short when CCI crosses above +100 in downtrend with volume confirmation.
# Exits when CCI returns to neutral zone (-50 to +50).
# Uses 1-day EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4H_CCI_MeanReversion_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate CCI(20) on 4h chart
    period = 20
    tp = (high + low + close) / 3.0  # Typical Price
    
    # Moving average of TP
    ma_tp = np.zeros_like(tp)
    ma_tp[period-1] = np.mean(tp[:period])
    for i in range(period, len(tp)):
        ma_tp[i] = ma_tp[i-1] + (tp[i] - tp[i-period]) / period
    
    # Mean deviation
    md = np.zeros_like(tp)
    for i in range(period-1, len(tp)):
        md[i] = np.mean(np.abs(tp[i-period+1:i+1] - ma_tp[i]))
    
    # CCI calculation
    cci = np.where(md != 0, (tp - ma_tp) / (0.015 * md), 0)
    
    # Volume average (20-period)
    vol_ma = np.zeros_like(volume)
    vol_ma[period-1] = np.mean(volume[:period])
    for i in range(period, len(volume)):
        vol_ma[i] = vol_ma[i-1] + (volume[i] - volume[i-period]) / period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 50)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(cci[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # CCI signals
        cci_below_neg100 = cci[i] < -100
        cci_above_pos100 = cci[i] > 100
        cci_below_neg50 = cci[i] < -50
        cci_above_pos50 = cci[i] > 50
        cci_cross_below_neg100 = cci_below_neg100 and (cci[i-1] >= -100)
        cci_cross_above_pos100 = cci_above_pos100 and (cci[i-1] <= 100)
        cci_return_to_neutral = cci_below_neg50 and cci_above_pos50
        
        if position == 0:
            # Long entry: CCI crosses below -100 in uptrend with volume spike
            if (cci_cross_below_neg100 and 
                price_above_ema and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: CCI crosses above +100 in downtrend with volume spike
            elif (cci_cross_above_pos100 and 
                  price_below_ema and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI returns to neutral zone
            if cci_return_to_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI returns to neutral zone
            if cci_return_to_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals