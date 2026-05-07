#!/usr/bin/env python3
name = "1d_Trix_Volume_Signal_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # TRIX calculation: EMA(EMA(EMA(close, period), period), period) then % change
    period = 15
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix = ema3.pct_change() * 100  # percentage change
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike detection (2x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period*3, 20)  # Need enough data for TRIX and volume
    
    for i in range(start_idx, n):
        if (np.isnan(trix.iloc[i]) if hasattr(trix, 'iloc') else np.isnan(trix[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix.iloc[i] if hasattr(trix, 'iloc') else trix[i]
        trix_prev = trix.iloc[i-1] if hasattr(trix, 'iloc') else trix[i-1]
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        weekly_uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
        weekly_downtrend = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume in weekly uptrend
            if trix_prev <= 0 and trix_val > 0 and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume in weekly downtrend
            elif trix_prev >= 0 and trix_val < 0 and vol_condition and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or weekly trend changes
            if trix_val < 0 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or weekly trend changes
            if trix_val > 0 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (triple exponential average) zero crosses with volume confirmation and weekly trend filter
# - TRIX filters out insignificant price movements and shows momentum
# - Zero line crosses indicate momentum shifts
# - Volume confirmation (2x average) ensures strong participation
# - Weekly EMA20 trend filter ensures alignment with higher timeframe trend
# - Works in bull markets (long on zero cross up in uptrend) and bear markets (short on zero cross down in downtrend)
# - Exit when TRIX reverses or weekly trend changes
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - TRIX is less commonly used than MACD/RSI, offering unique signal characteristics
# - Weekly trend filter reduces whipsaws vs same-timeframe signals
# - Aims for 80-200 total trades over 4 years (20-50/year) to stay within limits