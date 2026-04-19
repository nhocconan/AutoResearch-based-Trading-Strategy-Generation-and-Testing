#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend and 1d momentum confirmation
# - 4h EMA(34) defines trend direction (long when price > EMA34, short when price < EMA34)
# - 1d ROC(10) for momentum confirmation (positive for long, negative for short)
# - 1h Williams %R(14) for entry timing: long when %R < -80 in uptrend, short when %R > -20 in downtrend
# - Exit on opposite %R extreme or trend reversal
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods
# - Position size: 0.20 (20%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "1h_EMA34_WilliamsR_1dROC_v1"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(34) for trend direction
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for momentum confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ROC(10) for momentum
    roc_10_1d = pd.Series(df_1d['close'].values).pct_change(10).values
    roc_10_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_10_1d)
    
    # 1h Williams %R(14) for entry timing
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if np.isnan(ema_34_4h_aligned[i]) or np.isnan(roc_10_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Look for long entry: uptrend (price > 4h EMA34) + positive momentum + oversold Williams %R
            if close[i] > ema_34_4h_aligned[i] and roc_10_1d_aligned[i] > 0 and williams_r[i] < -80:
                signals[i] = 0.20
                position = 1
            # Look for short entry: downtrend (price < 4h EMA34) + negative momentum + overbought Williams %R
            elif close[i] < ema_34_4h_aligned[i] and roc_10_1d_aligned[i] < 0 and williams_r[i] > -20:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought Williams %R or trend reversal
            if williams_r[i] > -20 or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit on oversold Williams %R or trend reversal
            if williams_r[i] < -80 or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals