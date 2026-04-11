#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Strategy: Daily Camarilla pivot breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. A break above H4
# with weekly uptrend (price > weekly EMA20) and volume confirmation signals bullish momentum.
# Break below L4 with weekly downtrend (price < weekly EMA20) and volume confirmation signals
# bearish momentum. Designed for low trade frequency (~10-30/year) to minimize fee drag.
# Works in bull markets via breakout continuation and bear markets via breakdown continuation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily high, low, close for Camarilla calculation
    # Using previous day's OHLC to calculate today's levels (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current bar's high as previous (no prior data)
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels calculation
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Using previous day's values
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_h4[i]) or \
           np.isnan(camarilla_l4[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        # Long: Close > H4 (breakout) + weekly uptrend + volume confirmation
        if close[i] > camarilla_h4[i] and weekly_uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Close < L4 (breakdown) + weekly downtrend + volume confirmation
        elif close[i] < camarilla_l4[i] and weekly_downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout/breakdown (reversion to mean)
        elif position == 1 and close[i] < camarilla_l4[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > camarilla_h4[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals