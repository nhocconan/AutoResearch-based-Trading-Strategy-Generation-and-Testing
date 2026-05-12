#!/usr/bin/env python3
"""
1d_RSI2_Recovery_With_Volume_Confirmation
Hypothesis: On 1d timeframe, buy when RSI(2) < 10 (deep oversold) with volume > 1.5x average and price above 200-day SMA for trend filter; sell when RSI(2) > 90 (overbought) with volume > 1.5x average and price below 200-day SMA. Uses weekly ADX(14) > 20 to ensure we only trade in trending markets, avoiding whipsaws in ranges. Targets 10-25 trades per year to minimize fee drag and improve generalization in both bull and bear markets.
"""

name = "1d_RSI2_Recovery_With_Volume_Confirmation"
timeframe = "1d"
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

    # Get 1d data for RSI(2) and SMA(200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate RSI(2) on daily close
    def calculate_rsi(prices, period):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_2 = calculate_rsi(close_1d, 2)

    # Calculate 200-day SMA for trend filter
    sma_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values

    # Get weekly data for ADX(14) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate ADX(14) on weekly data
    def calculate_adx(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        def smooth_wilder(arr, period):
            result = np.zeros_like(arr)
            if len(arr) < period:
                return result
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        tr_smooth = smooth_wilder(tr, period)
        plus_dm_smooth = smooth_wilder(plus_dm, period)
        minus_dm_smooth = smooth_wilder(minus_dm, period)
        
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                      0)
        adx = smooth_wilder(dx, period)
        return adx

    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)

    # Align all indicators to 1d timeframe
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_2_aligned[i]) or np.isnan(sma_200_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI(2) < 10 (oversold) + price > SMA200 (uptrend) + volume spike + ADX > 20
            if (rsi_2_aligned[i] < 10 and 
                close[i] > sma_200_aligned[i] and 
                volume[i] > vol_avg_20_aligned[i] * 1.5 and
                adx_1w_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI(2) > 90 (overbought) + price < SMA200 (downtrend) + volume spike + ADX > 20
            elif (rsi_2_aligned[i] > 90 and 
                  close[i] < sma_200_aligned[i] and 
                  volume[i] > vol_avg_20_aligned[i] * 1.5 and
                  adx_1w_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI(2) > 50 (neutral) OR price < SMA200 OR ADX weakens
            if rsi_2_aligned[i] > 50 or close[i] < sma_200_aligned[i] or adx_1w_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI(2) < 50 (neutral) OR price > SMA200 OR ADX weakens
            if rsi_2_aligned[i] < 50 or close[i] > sma_200_aligned[i] or adx_1w_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals