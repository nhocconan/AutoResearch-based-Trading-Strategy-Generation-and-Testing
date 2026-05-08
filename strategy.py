#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_RSI_Trend_Scalper"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Daily ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed averages
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smma(tr, 14)
    plus_di = 100 * smma(plus_dm, 14) / atr
    minus_di = 100 * smma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smma(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Hourly RSI for entry timing
    rsi_period = 14
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period] = np.nanmean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.nanmean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, rsi_period+1)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i]) or not session_mask[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + strong trend + RSI oversold bounce
            long_cond = (close[i] > ema200_1d_aligned[i]) and \
                       (adx_aligned[i] > 25) and \
                       (rsi[i] < 35)
            
            # Short: Downtrend + strong trend + RSI overbought bounce
            short_cond = (close[i] < ema200_1d_aligned[i]) and \
                        (adx_aligned[i] > 25) and \
                        (rsi[i] > 65)
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Trend weakness or RSI overbought
            if (close[i] <= ema200_1d_aligned[i]) or (rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Trend weakness or RSI oversold
            if (close[i] >= ema200_1d_aligned[i]) or (rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h scalper using daily trend filter (EMA200 + ADX>25) for direction,
# hourly RSI for entry timing (oversold/overbought bounces), and session filter (08-20 UTC)
# to avoid low-liquidity hours. Works in both bull and bear markets by following
# the higher-timeframe trend while using mean-reversion entries on the lower timeframe.
# Target: 80-150 total trades over 4 years = 20-38/year to balance opportunity and cost.