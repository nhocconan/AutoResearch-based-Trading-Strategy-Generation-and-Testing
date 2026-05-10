#!/usr/bin/env python3
# 1h_Liquidity_Sweep_4hTrend_Volume
# Hypothesis: In 1h timeframe, price often sweeps liquidity (equal highs/lows) before reversing.
# We use 4h trend as filter (EMA50) and volume confirmation to avoid false signals.
# Liquidity sweep identified when price makes new 1h high/low but closes back inside
# previous hour's range, indicating stop hunt. Trades only in direction of 4h trend.
# Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year.

name = "1h_Liquidity_Sweep_4hTrend_Volume"
timeframe = "1h"
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
    open_time = prices['open_time']
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation (24-period MA on 1h = 1 day)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA50 (50), volume MA (24), and at least 2 bars for liquidity check
    start_idx = max(50, 24, 2)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend filter
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Liquidity sweep detection (1-hour timeframe)
        # Bullish sweep: new 1h low but closes back inside previous hour's range
        bearish_sweep = (
            low[i] < low[i-1] and  # new low
            close[i] > low[i-1] and  # closes above previous low
            close[i] < high[i-1]  # closes below previous high (inside range)
        )
        # Bearish sweep: new 1h high but closes back inside previous hour's range
        bullish_sweep = (
            high[i] > high[i-1] and  # new high
            close[i] < high[i-1] and  # closes below previous high
            close[i] > low[i-1]  # closes above previous low (inside range)
        )
        
        if position == 0:
            # Long entry: 4h uptrend + bullish sweep + volume
            if uptrend and bullish_sweep and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: 4h downtrend + bearish sweep + volume
            elif downtrend and bearish_sweep and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or opposite sweep occurs
            if not uptrend or bearish_sweep:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or opposite sweep occurs
            if not downtrend or bullish_sweep:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals