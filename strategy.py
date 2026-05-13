#!/usr/bin/env python3
# 1h_Combined_Trend_Momentum_Strategy
# Hypothesis: Combine 4h trend direction with 1h momentum and volume confirmation to capture medium-term moves while avoiding whipsaws.
# Long when: 4h EMA21 > EMA50 (uptrend), 1h RSI > 55 (momentum), and volume > 1.5x 20-period average (confirmation).
# Short when: 4h EMA21 < EMA50 (downtrend), 1h RSI < 45 (momentum), and volume > 1.5x 20-period average.
# Uses 4h for trend direction (reducing false signals) and 1h for entry timing.
# Includes session filter (08-20 UTC) to avoid low-volume Asian session noise.
# Position size fixed at 0.20 to manage risk and reduce churn.
# Designed to work in both bull (trend following) and bear (mean reversion within trend) markets.

name = "1h_Combined_Trend_Momentum_Strategy"
timeframe = "1h"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMAs for trend: EMA21 > EMA50 = uptrend, EMA21 < EMA50 = downtrend
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after RSI warmup
        # Skip if any required value is NaN
        if (np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h uptrend + 1h bullish momentum + volume confirmation
            if (ema21_4h_aligned[i] > ema50_4h_aligned[i] and 
                rsi[i] > 55 and 
                volume_confirmed[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + 1h bearish momentum + volume confirmation
            elif (ema21_4h_aligned[i] < ema50_4h_aligned[i] and 
                  rsi[i] < 45 and 
                  volume_confirmed[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h trend turns down OR momentum fades
            if (ema21_4h_aligned[i] < ema50_4h_aligned[i] or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend turns up OR momentum fades
            if (ema21_4h_aligned[i] > ema50_4h_aligned[i] or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals