#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + volume spike + chop regime filter
# KAMA adapts to market noise, reducing whipsaw in ranging markets. RSI(2) captures short-term extremes.
# Volume confirmation ensures breakout validity. Chop filter (CHOP > 61.8) triggers mean reversion in ranging markets.
# Designed for 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
# Works in bull markets via KAMA uptrend + RSI(2) pullback longs, and in bear markets via KAMA downtrend + RSI(2) bounces shorts.

name = "1d_KAMA_RSI2_VolumeSpike_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC) to avoid datetime64 issues
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Chopiness Index on 1w
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros(len(close_arr))
        tr = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = np.abs(high_arr[i] - close_arr[i-1])
            lc = np.abs(low_arr[i] - close_arr[i-1])
            tr[i] = max(hl, hc, lc)
        atr[0] = tr[0]
        for i in range(1, len(atr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
        sum_atr = np.zeros(len(close_arr))
        for i in range(period-1, len(close_arr)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        max_high = np.zeros(len(close_arr))
        min_low = np.zeros(len(close_arr))
        for i in range(period-1, len(close_arr)):
            max_high[i] = np.max(high_arr[i-period+1:i+1])
            min_low[i] = np.min(low_arr[i-period+1:i+1])
        chop = np.full(len(close_arr), np.nan)
        for i in range(period-1, len(close_arr)):
            if max_high[i] - min_low[i] != 0:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop_1w = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate KAMA on 1d
    def calculate_kama(close_arr, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close_arr, n=period))
        volatility = np.sum(np.abs(np.diff(close_arr)), axis=1) if len(close_arr) > 1 else np.array([0])
        volatility = np.concatenate([[np.sum(np.abs(np.diff(close_arr[:period])))], volatility[1:]]) if len(close_arr) > period else np.array([np.sum(np.abs(np.diff(close_arr)))])
        er = np.zeros(len(close_arr))
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = np.zeros(len(close_arr))
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros(len(close_arr))
        kama[0] = close_arr[0]
        for i in range(1, len(close_arr)):
            kama[i] = kama[i-1] + sc[i] * (close_arr[i] - kama[i-1])
        return kama
    
    kama_1d = calculate_kama(close, 10, 2, 30)
    kama_1d_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}, index=prices.index), kama_1d)
    
    # Calculate RSI(2) on 1d
    def calculate_rsi(close_arr, period=2):
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros(len(close_arr))
        avg_loss = np.zeros(len(close_arr))
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close_arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.zeros(len(close_arr))
        rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
        rsi = np.zeros(len(close_arr))
        rsi[avg_loss != 0] = 100 - (100 / (1 + rs[avg_loss != 0]))
        rsi[avg_loss == 0] = 100
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    rsi_2_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}, index=prices.index), rsi_2)
    
    # Volume confirmation: 20-period EMA on 1d
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_2_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        chop_condition = chop_1w_aligned[i] > 61.8  # Range regime
        
        if position == 0:
            # Long: KAMA uptrend + RSI(2) oversold + volume spike in chop
            if close[i] > kama_1d_aligned[i] and rsi_2_aligned[i] < 10 and volume_spike and chop_condition:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI(2) overbought + volume spike in chop
            elif close[i] < kama_1d_aligned[i] and rsi_2_aligned[i] > 90 and volume_spike and chop_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI(2) overbought or trend change
            if rsi_2_aligned[i] > 70 or close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI(2) oversold or trend change
            if rsi_2_aligned[i] < 30 or close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals