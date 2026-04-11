# This strategy targets 6h timeframe with weekly trend bias and daily pivot structure.
# Hypothesis: Weekly trend (via EMA200) filters direction, while daily Camarilla pivots provide high-probability reversal zones.
# Long only in weekly uptrend at S1/S2 support with bullish reversal candle.
# Short only in weekly downtrend at R3/R4 resistance with bearish reversal candle.
# Designed for low turnover (target: 15-35 trades/year) with high win rate in both bull and bear markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_daily_camarilla_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === WEEKLY TREND FILTER (applied once before loop) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === DAILY CAMARILLA PIVOTS (applied once before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # H = high, L = low, C = close of previous day
    H = np.roll(high_1d, 1)
    L = np.roll(low_1d, 1)
    C = np.roll(close_1d, 1)
    
    # First value will be NaN due to roll, that's expected
    P = (H + L + C) / 3
    range_hl = H - L
    
    R3 = C + (range_hl * 1.1 / 2)
    R4 = C + (range_hl * 1.1)
    S3 = C - (range_hl * 1.1 / 2)
    S4 = C - (range_hl * 1.1)
    
    # Align to 6h timeframe (wait for daily close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Bullish reversal candle: close > open and close > previous close
        bullish_reversal = (close[i] > prices['open'].iloc[i]) and (i > 0 and close[i] > close[i-1])
        # Bearish reversal candle: close < open and close < previous close
        bearish_reversal = (close[i] < prices['open'].iloc[i]) and (i > 0 and close[i] < close[i-1])
        
        # Entry conditions
        long_setup = weekly_uptrend and (
            (close[i] <= S3_aligned[i] * 1.005) or  # Near S3
            (close[i] <= S4_aligned[i] * 1.01)      # Near S4
        ) and bullish_reversal
        
        short_setup = weekly_downtrend and (
            (close[i] >= R3_aligned[i] * 0.995) or  # Near R3
            (close[i] >= R4_aligned[i] * 0.99)      # Near R4
        ) and bearish_reversal
        
        # Exit conditions: opposite reversal or midpoint reversion
        long_exit = bearish_reversal or (close[i] >= (S3_aligned[i] + S4_aligned[i]) / 2 * 1.02)
        short_exit = bullish_reversal or (close[i] <= (R3_aligned[i] + R4_aligned[i]) / 2 * 0.98)
        
        # Priority: entry > exit > hold
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals