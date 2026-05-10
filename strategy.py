#!/usr/bin/env python3
# 1d_Weekly_Keltner_Channel_Breakout_v1
# Hypothesis: Use weekly Keltner Channel breakouts on daily timeframe with volatility filter.
# In bull markets: break above upper KC signals momentum continuation.
# In bear markets: break below lower KC signals continuation of downtrend.
# Weekly timeframe reduces noise, daily provides timely entries. Volatility filter (ATR) avoids chop.
# Targets 10-20 trades/year to minimize fee drag. Works across BTC, ETH, SOL.

name = "1d_Weekly_Keltner_Channel_Breakout_v1"
timeframe = "1d"
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
    
    # Get weekly data for Keltner Channel calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate ATR for Keltner Channel (using weekly data)
    tr1 = df_weekly['high'] - df_weekly['low']
    tr2 = np.abs(df_weekly['high'] - np.roll(df_weekly['close'], 1))
    tr3 = np.abs(df_weekly['low'] - np.roll(df_weekly['close'], 1))
    tr1.iloc[0] = 0
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_weekly = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate weekly EMA (middle of KC)
    ema_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate upper and lower Keltner Channel bands
    kc_upper = ema_weekly + (atr_weekly * 2.0)
    kc_lower = ema_weekly - (atr_weekly * 2.0)
    
    # Align weekly KC levels to daily timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_weekly, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_weekly, kc_lower)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily ATR for volatility filter
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr1_d[0] = 0
    tr2_d[0] = 0
    tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_daily = pd.Series(tr_d).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Warmup for weekly EMA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(atr_daily[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        volatility_ok = atr_daily[i] > 0
        
        if position == 0:
            # Long entry: price breaks above upper KC with adequate volatility
            if close[i] > kc_upper_aligned[i] and volatility_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower KC with adequate volatility
            elif close[i] < kc_lower_aligned[i] and volatility_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below weekly EMA (middle of KC)
            if close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above weekly EMA (middle of KC)
            if close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals