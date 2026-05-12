#!/usr/bin/env python3
"""
1h VWAP + RSI + HTF Trend Filter
Hypothesis: Use 4h trend direction (via EMA) and 1d market regime (via ADX) to filter 1h VWAP mean-reversion entries.
In trending markets (ADX > 25), trade pullbacks to VWAP in direction of 4h EMA. In ranging markets (ADX < 20), fade extremes.
This reduces false signals and captures both trending and ranging behavior with low trade frequency.
"""
name = "1h_VWAP_RSI_HTFTrend"
timeframe = "1h"
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
    
    # === VWAP (session-based, reset daily) ===
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    # Reset at midnight UTC (00:00) - new session
    dates = pd.to_datetime(prices['open_time']).date
    vwap_reset = np.where(dates != np.roll(dates, 1), 0, 1)  # 1 if same day, 0 if new day
    vwap_reset[0] = 1  # first bar always start of session
    cum_pv = np.cumsum(pv * vwap_reset)
    cum_vol = np.cumsum(volume * vwap_reset)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === ADX (14) for 1d regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h EMA for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === Session filter: 08:00-20:00 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        ema_4h_val = ema_4h_aligned[i]
        price = close[i]
        vwap_val = vwap[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Enter long: ADX > 25 (trending) + price > 4h EMA (uptrend) + price < VWAP (pullback) + RSI < 40
            # Enter short: ADX > 25 (trending) + price < 4h EMA (downtrend) + price > VWAP (pullback) + RSI > 60
            if adx_val > 25:
                if price > ema_4h_val and price < vwap_val and rsi_val < 40:
                    signals[i] = 0.20
                    position = 1
                elif price < ema_4h_val and price > vwap_val and rsi_val > 60:
                    signals[i] = -0.20
                    position = -1
            # Ranging market: ADX < 20, fade extremes
            elif adx_val < 20:
                # Long when price significantly below VWAP and RSI oversold
                if price < vwap_val * 0.995 and rsi_val < 30:
                    signals[i] = 0.20
                    position = 1
                # Short when price significantly above VWAP and RSI overbought
                elif price > vwap_val * 1.005 and rsi_val > 70:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price > VWAP (reversion complete) OR RSI > 60 (overbought) OR ADX < 20 (range) + price > VWAP
            if price > vwap_val or rsi_val > 60 or (adx_val < 20 and price > vwap_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price < VWAP (reversion complete) OR RSI < 40 (oversold) OR ADX < 20 (range) + price < VWAP
            if price < vwap_val or rsi_val < 40 or (adx_val < 20 and price < vwap_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals