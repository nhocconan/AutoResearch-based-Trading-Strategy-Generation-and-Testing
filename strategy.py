#!/usr/bin/env python3
"""
1d_WeeklyKeltnerChannel_Breakout_Trend
Hypothesis: On 1d timeframe, price breaking above/below weekly Keltner Channel (ATR-based volatility band),
confirmed by weekly trend (price above/below weekly EMA50) and volume surge, captures trend continuation
in both bull and bear markets. Keltner Channels adapt to volatility, providing dynamic support/resistance.
Target: 10-25 trades/year per symbol.
"""

name = "1d_WeeklyKeltnerChannel_Breakout_Trend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Keltner Channel
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly trend: EMA50
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = weekly_close > ema50_weekly
    weekly_downtrend = weekly_close < ema50_weekly
    
    # Weekly ATR for Keltner Channel (20-period)
    tr = np.maximum(
        weekly_high[1:] - weekly_low[1:],
        np.maximum(
            np.abs(weekly_high[1:] - weekly_close[:-1]),
            np.abs(weekly_low[1:] - weekly_close[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])  # align length
    atr20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly Keltner Channel: EMA20 ± 2*ATR
    ema20_weekly = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20_weekly + 2 * atr20
    kc_lower = ema20_weekly - 2 * atr20
    
    # Align weekly indicators to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_downtrend)
    kc_upper_aligned = align_htf_to_ltf(prices, df_weekly, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_weekly, kc_lower)
    
    # Volume confirmation: today's volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned weekly values
        weekly_uptrend = weekly_uptrend_aligned[i]
        weekly_downtrend = weekly_downtrend_aligned[i]
        kc_upper = kc_upper_aligned[i]
        kc_lower = kc_lower_aligned[i]
        vol_surge = volume_surge[i]
        
        if position == 0:
            # LONG: weekly uptrend + price breaks above KC upper + volume surge
            if weekly_uptrend and close[i] > kc_upper and vol_surge:
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend + price breaks below KC lower + volume surge
            elif weekly_downtrend and close[i] < kc_lower and vol_surge:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below KC middle (EMA20) or trend reversal
            if close[i] < ema20_weekly[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above KC middle or trend reversal
            if close[i] > ema20_weekly[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals