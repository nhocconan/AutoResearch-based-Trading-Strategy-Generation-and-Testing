#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend (EMA21) and 1d momentum (ROC10) with volume confirmation
# Uses 4h for trend direction, 1d for momentum strength, 1h for entry timing with volume filter
# Designed to work in both bull (trend following) and bear (momentum reversals) markets
# Target: 15-30 trades/year to minimize fee drag while capturing meaningful moves

name = "1h_4hEMA_1dROC_VolumeFilter_V1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for momentum
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA21 for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1d ROC10 for momentum
    roc_period = 10
    roc_1d = np.full_like(close_1d, np.nan)
    for i in range(roc_period, len(close_1d)):
        if close_1d[i - roc_period] != 0:
            roc_1d[i] = ((close_1d[i] - close_1d[i - roc_period]) / close_1d[i - roc_period]) * 100
    
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    # Calculate 1h volume spike (volume > 1.5 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(roc_1d_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
            
        # Entry conditions:
        # Long: 4h uptrend (price > EMA21) + positive 1d momentum + volume spike
        # Short: 4h downtrend (price < EMA21) + negative 1d momentum + volume spike
        trend_up = close[i] > ema_21_4h_aligned[i]
        trend_down = close[i] < ema_21_4h_aligned[i]
        mom_up = roc_1d_aligned[i] > 0
        mom_down = roc_1d_aligned[i] < 0
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long entry
            if trend_up and mom_up and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry
            elif trend_down and mom_down and vol_confirm:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long exit: trend breaks down or momentum turns negative
            if not trend_up or not mom_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short exit: trend breaks up or momentum turns positive
            if not trend_down or not mom_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals