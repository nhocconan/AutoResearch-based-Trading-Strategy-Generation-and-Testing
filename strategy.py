#!/usr/bin/env python3
# 6h_ConnorRSI_MeanReversion
# Hypothesis: ConnorRSI (RSI(3) + RSI(2) streak + PercentRank(100))/3 on 6h timeframe.
# Long when ConnorRSI < 15 and price above 1d EMA200 (bullish regime).
# Short when ConnorRSI > 85 and price below 1d EMA200 (bearish regime).
# Exit when ConnorRSI crosses above 70 (long) or below 30 (short).
# Designed for low trade frequency (20-60 total trades over 4 years) with high win rate.
# Works in bull markets via trend alignment and in bear markets via mean reversion against trend.

name = "6h_ConnorRSI_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period):
    """Calculate RSI with given period."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def stoch_rsi(rsi_series, period):
    """Calculate StochRSI (percentile rank) of RSI series."""
    return pd.Series(rsi_series).rolling(window=period, min_periods=period).apply(
        lambda x: (x.iloc[-1] - np.min(x)) / (np.max(x) - np.min(x) + 1e-10) * 100, raw=False
    ).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values

    # Calculate ConnorRSI components on 6h close
    rsi_3 = rsi(close, 3)
    rsi_2 = rsi(close, 2)
    
    # Calculate RSI streak (consecutive up/down days)
    delta = np.diff(close, prepend=close[0])
    up_days = np.where(delta > 0, 1, 0)
    down_days = np.where(delta < 0, 1, 0)
    streak_up = np.where(up_days, np.concatenate([[0], np.cumsum(up_days[:-1]) * up_days]), 0)
    streak_down = np.where(down_days, np.concatenate([[0], np.cumsum(down_days[:-1]) * down_days]), 0)
    rsi_streak = np.where(streak_up > 0, streak_up, -streak_down)
    rsi_streak_rsi = rsi(np.where(rsi_streak > 0, close, np.nan), 2)  # RSI of streak
    rsi_streak_rsi = np.where(~np.isnan(rsi_streak_rsi), rsi_streak_rsi, 50)  # fill NaN with 50
    
    # Percent Rank of RSI(3) over 100 periods
    percent_rank = stoch_rsi(rsi_3, 100)
    
    # ConnorRSI = (RSI(3) + RSI(streak) + PercentRank) / 3
    connor_rsi = (rsi_3 + rsi_streak_rsi + percent_rank) / 3

    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after 100 for percent rank calculation
        # Skip if any required value is NaN
        if (np.isnan(connor_rsi[i]) or np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: ConnorRSI < 15 (oversold) + price above 1d EMA200 (bullish regime)
            if connor_rsi[i] < 15 and close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: ConnorRSI > 85 (overbought) + price below 1d EMA200 (bearish regime)
            elif connor_rsi[i] > 85 and close[i] < ema_200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ConnorRSI crosses above 70 (overbought)
            if connor_rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ConnorRSI crosses below 30 (oversold)
            if connor_rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals