#!/usr/bin/env python3
# 12h_Adaptive_Kelly_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaking out of Camarilla R3/S3 levels on 12h with 1d trend and volume confirmation, using Kelly criterion for position sizing (capped at 0.30) to optimize risk-adjusted returns. Works in both bull and bear markets by filtering with 1d trend and avoiding whipsaws via volume confirmation. Uses adaptive position sizing to reduce drawdowns during high volatility periods.

name = "12h_Adaptive_Kelly_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Calculate ATR for volatility normalization (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # 1d EMA34 for trend filter (load once, align)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.8x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    # Kelly criterion components: win rate and win/loss ratio (estimated from recent performance)
    # Using 60-period lookback for adaptive sizing
    returns = np.diff(close, prepend=close[0]) / close
    wins = np.maximum(returns, 0)
    losses = np.maximum(-returns, 0)
    
    win_rate = pd.Series(wins).rolling(window=60, min_periods=30).mean().values
    avg_win = pd.Series(wins).rolling(window=60, min_periods=30).mean().values
    avg_loss = pd.Series(losses).rolling(window=60, min_periods=30).mean().values
    win_loss_ratio = np.where(avg_loss > 0, avg_win / avg_loss, 1.0)
    
    # Kelly fraction: f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    kelly_fraction = np.where(
        (win_loss_ratio > 0) & (win_rate > 0),
        (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio,
        0.0
    )
    # Cap Kelly at 0.30 and ensure non-negative
    kelly_fraction = np.clip(kelly_fraction, 0.0, 0.30)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_30[i]) or 
            np.isnan(kelly_fraction[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Calculate Camarilla levels for current 12h bar (using previous bar's range)
        if i > 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            if range_val > 0:
                camarilla_multiplier = 1.1 / 12
                r3 = prev_close + range_val * camarilla_multiplier * 4
                s3 = prev_close - range_val * camarilla_multiplier * 4
            else:
                r3 = prev_close
                s3 = prev_close
        else:
            r3 = close[0]
            s3 = close[0]

        if position == 0:
            # LONG: Close above R3 + 1d EMA34 uptrend + volume spike
            if (close[i] > r3 and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.8):
                # Use Kelly fraction for position sizing, capped at 0.30
                signal_size = min(kelly_fraction[i], 0.30)
                signals[i] = signal_size
                position = 1
            # SHORT: Close below S3 + 1d EMA34 downtrend + volume spike
            elif (close[i] < s3 and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.8):
                # Use Kelly fraction for position sizing, capped at 0.30
                signal_size = min(kelly_fraction[i], 0.30)
                signals[i] = -signal_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below 1d EMA34 or volatility drop
            if close[i] < ema34_1d_aligned[i] or volume[i] < vol_avg_30[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = min(kelly_fraction[i], 0.30)
        elif position == -1:
            # EXIT SHORT: Close above 1d EMA34 or volatility drop
            if close[i] > ema34_1d_aligned[i] or volume[i] < vol_avg_30[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -min(kelly_fraction[i], 0.30)

    return signals