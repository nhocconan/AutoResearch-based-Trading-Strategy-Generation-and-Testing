#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_S1R1_Breakout_Trend"
timeframe = "1h"
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
    open_time = pd.to_datetime(prices['open_time'])
    hour = open_time.dt.hour.values
    
    # Session filter: 08-20 UTC
    in_session = (hour >= 8) & (hour <= 20)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Trend: EMA(50)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily Close for trend
    daily_close_prev = df_1d['close'].shift(1).values
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_prev)
    
    # Daily ATR-based bands (1.5 * ATR)
    atr_mult = 1.5
    upper_band = daily_close_aligned + (atr_mult * atr_14_1d_aligned)
    lower_band = daily_close_aligned - (atr_mult * atr_14_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Wait for EMA and ATR
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper band in 4h uptrend
            if close[i] > upper_band[i] and ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                signals[i] = 0.20
                position = 1
            # Short: price below lower band in 4h downtrend
            elif close[i] < lower_band[i] and ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below upper band or trend reversal
            if close[i] < upper_band[i] or ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above lower band or trend reversal
            if close[i] > lower_band[i] or ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h session-based ATR band breakout with 4h trend filter
# - Uses daily ATR(14) to set dynamic upper/lower bands (mean ± 1.5*ATR)
# - Entry only during 08-20 UTC (active session) to reduce noise
# - 4h EMA(50) trend filter ensures alignment with higher timeframe momentum
# - Long when price breaks above upper band in 4h uptrend
# - Short when price breaks below lower band in 4h downtrend
# - Exit when price returns to band or trend reverses
# - Position size 0.20 balances return and risk
# - Designed to work in both bull and bear markets via trend following
# - Target: 15-30 trades/year to avoid fee drag (max 120 total over 4 years)