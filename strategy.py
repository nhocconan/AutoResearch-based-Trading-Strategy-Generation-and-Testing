#!/usr/bin/env python3
# 1d_1W_Camarilla_R3S3_Breakout_Trend_Volume_Filtered
# Hypothesis: Breakout at weekly Camarilla R3/S3 levels with daily trend filter and volume confirmation.
# Uses weekly price structure for direction and daily timeframe for execution to reduce noise.
# Targets 15-25 trades/year to minimize fee drag while capturing major trend moves.

name = "1d_1W_Camarilla_R3S3_Breakout_Trend_Volume_Filtered"
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

    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate weekly RSI for overbought/oversold filter
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)

    # Calculate Camarilla R3 and S3 levels from previous weekly OHLC
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values

    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on weekly
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30

        if position == 0:
            # LONG: Break above weekly Camarilla R3 with bullish trend, volume confirmation, and RSI not overbought
            if (close[i] > camarilla_r3_aligned[i] and bullish_trend and volume_ok[i] and rsi_not_overbought):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Camarilla S3 with bearish trend, volume confirmation, and RSI not oversold
            elif (close[i] < camarilla_s3_aligned[i] and bearish_trend and volume_ok[i] and rsi_not_oversold):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns bearish or RSI overbought
            if close[i] < camarilla_r3_aligned[i] or not bullish_trend or rsi_1w_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns bullish or RSI oversold
            if close[i] > camarilla_s3_aligned[i] or not bearish_trend or rsi_1w_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals