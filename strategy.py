#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d (daily candles)
- HTF: 1w (weekly) for EMA50 trend alignment
- Williams %R(14) identifies overbought/oversold conditions
- Long when Williams %R crosses above -80 (exiting oversold) AND price > 1w EMA50 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when Williams %R crosses below -20 (exiting overbought) AND price < 1w EMA50 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when Williams %R returns to -50 (mean reversion center)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year) as per 1d timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Williams %R captures reversals in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Williams %R signals: crossing above -80 (long) or below -20 (short)
    williams_r_long_signal = williams_r > -80
    williams_r_short_signal = williams_r < -20
    williams_r_exit_signal = (williams_r > -50) & (williams_r < -50)  # Will be False, using -50 as exit level
    
    # For exit: when Williams %R crosses -50
    williams_r_above_50 = williams_r > -50
    williams_r_below_50 = williams_r < -50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # Need Williams %R(14), 1w EMA50, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (exiting oversold) AND uptrend AND volume confirmation
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (exiting overbought) AND downtrend AND volume confirmation
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0