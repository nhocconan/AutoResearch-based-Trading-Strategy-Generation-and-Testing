#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R1S1_Breakout_Trend_Confirm_v1
Hypothesis: On daily timeframe, enter long when price breaks above Camarilla R1 with weekly trend confirmation (price > weekly EMA50), enter short when price breaks below S1 with weekly trend confirmation (price < weekly EMA50). Use volume confirmation to avoid false breakouts. Designed for low trade frequency (~10-20 trades/year) by requiring multiple confluence factors. Works in bull markets via long breakouts and in bear markets via short breakdowns.
"""

name = "1d_Weekly_Camarilla_R1S1_Breakout_Trend_Confirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend confirmation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's H, L, C to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First day: use same day's values (will not trigger until second day anyway)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Weekly trend: EMA50 on weekly close
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Volume confirmation: volume > 1.5 * 20-day average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma20[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakout with volume confirmation and trend alignment
            bullish_breakout = (close[i] > R1[i]) and volume_confirm[i] and (close[i] > weekly_ema50_aligned[i])
            bearish_breakout = (close[i] < S1[i]) and volume_confirm[i] and (close[i] < weekly_ema50_aligned[i])
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Camarilla center (previous day's close) or trend reversal
            if position == 1:
                # Exit long: price < previous day's close or weekly trend turns bearish
                exit_signal = (close[i] < prev_close[i]) or (close[i] < weekly_ema50_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price > previous day's close or weekly trend turns bullish
                exit_signal = (close[i] > prev_close[i]) or (close[i] > weekly_ema50_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals