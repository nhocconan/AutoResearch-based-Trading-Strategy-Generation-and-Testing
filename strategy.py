#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d regime filter
# Long when 4h EMA(21) trending up, 1d ADX < 25 (range), and 1h RSI(2) crosses above 10
# Short when 4h EMA(21) trending down, 1d ADX < 25 (range), and 1h RSI(2) crosses below 90
# Uses 4h for trend direction, 1d for range regime, 1h for precise entry timing
# Designed for low trade frequency (target: 80-120 total over 4 years) to minimize fee drag
# Works in bull/bear by fading extreme 1h RSI in ranging markets with 4h trend alignment

name = "1h_rsi2_4hma_1dadx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA(21) for trend direction - calculated once before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_trend = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ADX(14) for regime detection (range vs trend) - calculated once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / (tr_ma + 1e-10)
    di_minus = 100 * dm_minus_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    adx_1d = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h RSI(2) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available or outside session
        if (np.isnan(ema_4h_trend[i]) or np.isnan(adx_1d[i]) or np.isnan(rsi[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI returns to neutral (50) or ADX > 25 (trending regime)
        if position == 1:  # long position
            if rsi[i] >= 50 or adx_1d[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] <= 50 or adx_1d[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in ranging market (ADX < 25) with extreme RSI
            # Long: RSI extremely oversold (<10) in ranging market + 4h uptrend
            if (adx_1d[i] < 25 and rsi[i] < 10 and ema_4h_trend[i] > ema_4h_trend[max(0, i-1)]):
                signals[i] = 0.20
                position = 1
            # Short: RSI extremely overbought (>90) in ranging market + 4h downtrend
            elif (adx_1d[i] < 25 and rsi[i] > 90 and ema_4h_trend[i] < ema_4h_trend[max(0, i-1)]):
                signals[i] = -0.20
                position = -1
    
    return signals