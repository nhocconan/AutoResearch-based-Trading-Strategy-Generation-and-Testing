#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_With_RSI_Filter
# Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) as trend filter and RSI(14) for mean-reversion entries.
# In trending markets (price > KAMA), buy dips when RSI < 30; in ranging markets, fade extremes at RSI < 30 or > 70.
# Uses 1-week ADX to distinguish trend (ADX > 25) from range (ADX < 20) to avoid whipsaws.
# Targets 10-25 trades/year by requiring confluence of trend, RSI extreme, and volume confirmation.

name = "1d_1w_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1-week ADX (14-period) for trend/range filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1w = smooth_wilder(tr, 14)
    plus_di_1w = 100 * smooth_wilder(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * smooth_wilder(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = smooth_wilder(dx_1w, 14)
    
    # Align 1w ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate KAMA on 1d close (ER=10, fastest=2, slowest=30)
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.concatenate([[np.nan], close[1:] - close[:-1]]))
        volatility = np.nansum(np.abs(np.concatenate([[np.nan], close[1:] - close[:-1]])), axis=0) if False else None
        # Proper volatility calculation: sum of absolute changes over er_period
        volatility = np.array([np.nansum(np.abs(close[i-er_period+1:i+1] - close[i-er_period:i])) 
                              if i >= er_period-1 else np.nan for i in range(len(close))])
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.full_like(close, np.nan)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc[i]) or np.isnan(kama_vals[i-1]):
                kama_vals[i] = kama_vals[i-1]
            else:
                kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, er_period=10, fast=2, slow=30)
    
    # Calculate RSI(14) on 1d close
    def rsi(close, period=14):
        delta = np.concatenate([[np.nan], close[1:] - close[:-1]])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        # Wilder smoothing
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        avg_gain[period] = np.nansum(gain[1:period+1]) / period
        avg_loss[period] = np.nansum(loss[1:period+1]) / period
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, period=14)
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine market regime using 1w ADX
            if adx_1w_aligned[i] > 25:  # Trending market
                # In uptrend (price > KAMA), buy dips when RSI oversold
                if close[i] > kama_vals[i] and rsi_vals[i] < 30 and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = 0.25
                    position = 1
                # In downtrend (price < KAMA), sell rallies when RSI overbought
                elif close[i] < kama_vals[i] and rsi_vals[i] > 70 and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market (ADX <= 25)
                # Fade extremes at RSI levels with volume confirmation
                if rsi_vals[i] < 30 and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi_vals[i] > 70 and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 or trend changes
            if rsi_vals[i] > 50 or (adx_1w_aligned[i] > 25 and close[i] < kama_vals[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 or trend changes
            if rsi_vals[i] < 50 or (adx_1w_aligned[i] > 25 and close[i] > kama_vals[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals