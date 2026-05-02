#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels (R1/S1) derived from daily OHLC provide strong intraday support/resistance
# Breakout above R1 or below S1 with volume confirmation indicates short-term momentum
# 4h EMA50 > 4h EMA200 ensures alignment with higher timeframe trend to avoid range-bound whipsaws
# Designed for 1h timeframe targeting 15-37 trades/year (60-150 total over 4 years)
# Uses discrete position sizing (0.20) to minimize fee churn and control drawdown
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Works in bull markets (breakout above R1 + 4h EMA50 > EMA200) and bear markets (breakout below S1 + 4h EMA50 < EMA200)

name = "1h_Camarilla_R1S1_Breakout_4hEMA_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter (EMA crossover) and HTF context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA200
        return np.zeros(n)
    
    # 4h EMA50 and EMA200 for trend filter
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # 1d data for Camarilla pivot levels (R1/S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on same day's OHLC)
    # Standard Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_r1 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    camarilla_s1 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (use same day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Moderate threshold for 1h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA crossover
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 with volume confirmation and uptrend
            if close[i] > camarilla_r1_aligned[i] and uptrend and volume_confirmation[i]:
                signals[i] = 0.20
                position = 1
            # Short: Breakout below S1 with volume confirmation and downtrend
            elif close[i] < camarilla_s1_aligned[i] and downtrend and volume_confirmation[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S1 (reversal) OR trend changes to downtrend
            if close[i] < camarilla_s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R1 (reversal) OR trend changes to uptrend
            if close[i] > camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals