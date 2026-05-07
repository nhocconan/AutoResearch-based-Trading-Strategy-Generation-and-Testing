#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend and Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1, S1
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 1h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if session[i]:
            if position == 0:
                # Long: Close breaks above R1 with volume spike in uptrend
                if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: Close breaks below S1 with volume spike in downtrend
                elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Exit: Close below S1 or trend turns down
                if close[i] < S1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit: Close above R1 or trend turns up
                if close[i] > R1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session: force flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout on 1h with 1d EMA34 trend filter and volume confirmation, active only during 08-20 UTC session.
# Long when price breaks above R1 (bullish breakout) with volume spike in 1d uptrend.
# Short when price breaks below S1 (bearish breakdown) with volume spike in 1d downtrend.
# Uses discrete position size (0.20) to minimize churn. Session filter reduces noise trades.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h, staying within HARD MAX of 200 total.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Volume spike (>1.8x average) ensures conviction behind the move.
# Session filter (08-20 UTC) avoids low-liquidity periods, improving signal quality.