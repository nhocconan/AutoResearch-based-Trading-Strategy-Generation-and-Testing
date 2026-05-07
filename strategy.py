#!/usr/bin/env python3
name = "6h_RSI2_Overbought_Oversold_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 6h RSI(2)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) in weekly uptrend with volume spike
            if rsi[i] < 10 and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and volume[i] > vol_ma_20[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) in weekly downtrend with volume spike
            elif rsi[i] > 90 and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and volume[i] > vol_ma_20[i] * 2.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI(2) > 50 (mean reversion) or trend change
            if rsi[i] > 50 or ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI(2) < 50 (mean reversion) or trend change
            if rsi[i] < 50 or ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s RSI(2) extreme readings with weekly trend filter and volume confirmation
# - RSI(2) < 10 indicates extreme oversold conditions (potential mean reversion long)
# - RSI(2) > 90 indicates extreme overbought conditions (potential mean reversion short)
# - Weekly EMA(34) trend filter ensures we only trade in direction of higher timeframe trend
# - Volume spike (2x 20-period average) confirms institutional interest at extremes
# - Works in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)
# - Exit when RSI returns to neutral (50) or trend changes
# - Position size 0.25 targets ~50-150 trades over 4 years (12-37/year) to avoid fee drag
# - RSI(2) is highly sensitive and captures short-term exhaustion moves
# - Weekly filter prevents counter-trend trading during strong moves
# - Volume confirmation reduces false signals from low-liquidity periods
# - Simple, robust logic with clear entry/exit conditions
# - Aims for 60-120 total trades over 4 years (15-30/year) to stay within limits