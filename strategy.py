#!/usr/bin/env python3
"""
1d_WeeklyKeltnerBreakout_TrendFilter
Hypothesis: Price breaking above/below weekly Keltner Channels (EMA20 ± 2*ATR) with daily EMA50 trend filter and volume confirmation captures sustained moves while filtering noise. Works in bull/bear by following daily trend direction. Uses weekly structure to reduce whipsaw and increase edge in trending markets.
"""

name = "1d_WeeklyKeltnerBreakout_TrendFilter"
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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 and ATR(14) for Keltner Channels
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # EMA20 weekly
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) weekly
    tr_weekly = np.maximum(
        high_weekly[1:] - low_weekly[1:],
        np.maximum(
            np.abs(high_weekly[1:] - close_weekly[:-1]),
            np.abs(low_weekly[1:] - close_weekly[:-1])
        )
    )
    tr_weekly = np.concatenate([[np.nan], tr_weekly])
    atr_14_weekly = pd.Series(tr_weekly).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Keltner Channels: EMA20 ± 2*ATR
    keltner_upper = ema_20_weekly + 2 * atr_14_weekly
    keltner_lower = ema_20_weekly - 2 * atr_14_weekly
    
    # Align Keltner Channels to daily timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_weekly, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_weekly, keltner_lower)
    
    # Daily EMA50 trend filter
    ema_50_daily = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_50_daily[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Keltner Upper + daily EMA50 uptrend + volume spike
            if (close[i] > keltner_upper_aligned[i] and 
                close[i] > ema_50_daily[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Keltner Lower + daily EMA50 downtrend + volume spike
            elif (close[i] < keltner_lower_aligned[i] and 
                  close[i] < ema_50_daily[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly Keltner Lower (reversal signal)
            if close[i] < keltner_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly Keltner Upper (reversal signal)
            if close[i] > keltner_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals