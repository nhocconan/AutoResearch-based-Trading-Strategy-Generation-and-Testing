#!/usr/bin/env python3
# 12h_Stochastic_RSI_Overbought_Oversold_RSI_Signal_1dTrend
# Hypothesis: Use Stochastic RSI to identify overbought/oversold conditions on 12h timeframe, confirmed by daily RSI and volume spike.
# Stochastic RSI captures momentum extremes; combined with daily RSI trend filter and volume confirmation provides high-probability entries.
# Works in bull (oversold bounces in uptrend) and bear (overbought reversals in downtrend).
# Low frequency due to strict Stochastic RSI thresholds and volume confirmation.

name = "12h_Stochastic_RSI_Overbought_Oversold_RSI_Signal_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for RSI trend filter and volume calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Daily RSI for trend filter
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_1d = calculate_rsi(df_1d['close'].values, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike: volume > 2.0 * 24-period average (2 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    # Stochastic RSI on 12h data
    def calculate_stoch_rsi(prices, rsi_period=14, stoch_period=14, k_period=3):
        # First calculate RSI
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        for i in range(rsi_period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        # Then calculate Stochastic of RSI
        stoch_rsi = np.full_like(prices, np.nan)
        for i in range(stoch_period-1, len(rsi)):
            window_rsi = rsi[i-stoch_period+1:i+1]
            min_rsi = np.min(window_rsi)
            max_rsi = np.max(window_rsi)
            if max_rsi - min_rsi != 0:
                stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
            else:
                stoch_rsi[i] = 50  # neutral when range is zero
        
        # Calculate %K (smoothed)
        k = np.full_like(prices, np.nan)
        for i in range(k_period-1, len(stoch_rsi)):
            if not np.isnan(stoch_rsi[i-k_period+1:i+1]).any():
                k[i] = np.mean(stoch_rsi[i-k_period+1:i+1])
        
        return k

    stoch_rsi_12h = calculate_stoch_rsi(close, 14, 14, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(stoch_rsi_12h[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Stochastic RSI oversold (<20) + daily RSI > 50 (uptrend) + volume spike
            if stoch_rsi_12h[i] < 20 and rsi_1d_aligned[i] > 50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Stochastic RSI overbought (>80) + daily RSI < 50 (downtrend) + volume spike
            elif stoch_rsi_12h[i] > 80 and rsi_1d_aligned[i] < 50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic RSI overbought (>80) OR daily RSI < 40 (weakening trend)
            if stoch_rsi_12h[i] > 80 or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Stochastic RSI oversold (<20) OR daily RSI > 60 (weakening trend)
            if stoch_rsi_12h[i] < 20 or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals