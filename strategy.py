#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + 1d RSI extremes + volume confirmation
# Long when KAMA direction is up (price > KAMA) AND 1d RSI < 30 (oversold) AND volume > 1.5x 20-period 1d volume SMA
# Short when KAMA direction is down (price < KAMA) AND 1d RSI > 70 (overbought) AND volume > 1.5x 20-period 1d volume SMA
# KAMA adapts to market noise, reducing false signals in choppy markets. RSI identifies extremes for mean reversion.
# Volume confirms conviction. Discrete position sizing (0.25) to control drawdown and fees.
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data once before loop for RSI and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: KAMA (10, 2, 30) ===
    close_4h = df_4h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_4h, n=10))
    volatility = np.sum(np.abs(np.diff(close_4h, n=1)), axis=0)[:len(change)]
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # start after first 10 periods
    for i in range(10, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # === 1d Indicator: RSI (14-period) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])  # align length
    # Align RSI to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (need 30 periods for KAMA + 14 for RSI + 20 for volume SMA)
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        vol_confirm = False
        if not np.isnan(vol_1d_aligned[i]):
            vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
            vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # === LONG CONDITIONS ===
        # Price > KAMA (uptrend) AND RSI < 30 (oversold) AND volume confirmation
        if (close[i] > kama_aligned[i]) and (rsi_aligned[i] < 30) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price < KAMA (downtrend) AND RSI > 70 (overbought) AND volume confirmation
        elif (close[i] < kama_aligned[i]) and (rsi_aligned[i] > 70) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_KAMA_1dRSI_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0