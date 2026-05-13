#!/usr/bin/env python3
# 4h_RSI_Divergence_4hTrend_VolumeConfirm
# Hypothesis: RSI divergence on 4h timeframe combined with 4h trend filter and volume confirmation
# captures high-probability reversals with low whipsaw. Works in both bull (bullish divergence at support)
# and bear (bearish divergence at resistance) markets. 4h trend ensures alignment with intermediate-term
# momentum, reducing false signals. Volume filter confirms reversal strength. Target: 20-40 trades/year.

name = "4h_RSI_Divergence_4hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter (using same timeframe as primary)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values

    # Calculate 4h RSI for divergence detection
    rsi_4h = calculate_rsi(close_4h, 14)
    
    # Calculate 4h EMA20 for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h indicators to lower timeframe (if needed, but here primary is 4h)
    # Since primary timeframe is 4h, we use the values directly
    rsi_4h_aligned = rsi_4h
    ema_4h_aligned = ema_4h

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # BULLISH DIVERGENCE: Price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 30:  # Need sufficient history
                # Check for price lower low
                if low[i] < low[i-10] and low[i-10] < low[i-20]:
                    # Check for RSI higher low
                    if not np.isnan(rsi_4h_aligned[i]) and not np.isnan(rsi_4h_aligned[i-10]) and not np.isnan(rsi_4h_aligned[i-20]):
                        if rsi_4h_aligned[i] > rsi_4h_aligned[i-10] and rsi_4h_aligned[i-10] > rsi_4h_aligned[i-20]:
                            bullish_div = True
            
            # BEARISH DIVERGENCE: Price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 30:
                # Check for price higher high
                if high[i] > high[i-10] and high[i-10] > high[i-20]:
                    # Check for RSI lower high
                    if not np.isnan(rsi_4h_aligned[i]) and not np.isnan(rsi_4h_aligned[i-10]) and not np.isnan(rsi_4h_aligned[i-20]):
                        if rsi_4h_aligned[i] < rsi_4h_aligned[i-10] and rsi_4h_aligned[i-10] < rsi_4h_aligned[i-20]:
                            bearish_div = True

            # LONG: Bullish divergence with volume spike and price above EMA (uptrend condition)
            if bullish_div and volume_spike[i] and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence with volume spike and price below EMA (downtrend condition)
            elif bearish_div and volume_spike[i] and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price crosses below EMA
            exit_signal = False
            if i >= 30:
                # Check for bearish divergence
                if high[i] > high[i-10] and high[i-10] > high[i-20]:
                    if not np.isnan(rsi_4h_aligned[i]) and not np.isnan(rsi_4h_aligned[i-10]) and not np.isnan(rsi_4h_aligned[i-20]):
                        if rsi_4h_aligned[i] < rsi_4h_aligned[i-10] and rsi_4h_aligned[i-10] < rsi_4h_aligned[i-20]:
                            exit_signal = True
            
            if exit_signal or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price crosses above EMA
            exit_signal = False
            if i >= 30:
                # Check for bullish divergence
                if low[i] < low[i-10] and low[i-10] < low[i-20]:
                    if not np.isnan(rsi_4h_aligned[i]) and not np.isnan(rsi_4h_aligned[i-10]) and not np.isnan(rsi_4h_aligned[i-20]):
                        if rsi_4h_aligned[i] > rsi_4h_aligned[i-10] and rsi_4h_aligned[i-10] > rsi_4h_aligned[i-20]:
                            exit_signal = True
            
            if exit_signal or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals