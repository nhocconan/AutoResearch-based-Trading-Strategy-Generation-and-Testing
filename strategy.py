#!/usr/bin/env python3

# 6h_1D_1W_Camarilla_R3S3_Breakout_Trend_Filter
# Hypothesis: Breakouts at Camarilla R3/S3 levels on 6h with 1d trend filter and 1w momentum confirmation.
# Uses daily trend and weekly momentum to filter breakouts, reducing false signals in choppy markets.
# Targets 15-25 trades/year per symbol (60-100 total over 4 years).

name = "6h_1D_1W_Camarilla_R3S3_Breakout_Trend_Filter"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Get 1w data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate 1w RSI for momentum filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)

    # Calculate 6h Camarilla levels from previous day
    # Using previous day's OHLC (1d data shifted by 1)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 6h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_val = prev_high_aligned - prev_low_aligned
    camarilla_r3 = prev_close_aligned + (range_val * 1.1 / 6)
    camarilla_s3 = prev_close_aligned - (range_val * 1.1 / 6)

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30

        if position == 0:
            # LONG: Break above Camarilla R3 with bullish trend and volume confirmation
            if (close[i] > camarilla_r3[i] and bullish_trend and 
                volume_ok[i] and rsi_not_overbought):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S3 with bearish trend and volume confirmation
            elif (close[i] < camarilla_s3[i] and bearish_trend and 
                  volume_ok[i] and rsi_not_oversold):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla H4 or trend turns bearish
            camarilla_h4 = prev_close_aligned[i] + (range_val[i] * 1.1 / 2)
            if (close[i] < camarilla_h4 or not bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla L4 or trend turns bullish
            camarilla_l4 = prev_close_aligned[i] - (range_val[i] * 1.1 / 2)
            if (close[i] > camarilla_l4 or not bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals