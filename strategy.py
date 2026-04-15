#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike
# Long when 1h RSI < 30 + 4h EMA50 > EMA200 (uptrend) + volume > 2x 20-period avg
# Short when 1h RSI > 70 + 4h EMA50 < EMA200 (downtrend) + volume > 2x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-30 trades/year.
# RSI mean reversion works in ranging markets; 4h EMA filter ensures we only trade with the higher timeframe trend.
# Volume spike confirms institutional participation, reducing false signals.

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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # === 4h Indicators: EMA50 and EMA200 (trend filter) ===
    close_4h = df_4h['close'].values
    
    # Calculate EMA50
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate EMA200
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Determine trend: 1 if EMA50 > EMA200 (uptrend), -1 if EMA50 < EMA200 (downtrend)
    trend_4h = np.where(ema50_4h > ema200_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === 1h Indicators: RSI(14) and Volume SMA(20) ===
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    period_rsi = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[period_rsi-1] = np.mean(gain[:period_rsi])
    avg_loss[period_rsi-1] = np.mean(loss[:period_rsi])
    
    for i in range(period_rsi, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period_rsi-1) + gain[i]) / period_rsi
        avg_loss[i] = (avg_loss[i-1] * (period_rsi-1) + loss[i]) / period_rsi
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume SMA for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 20)  # EMA200 and volume SMA
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 1h RSI < 30 (oversold)
        # 2. 4h trend = uptrend (EMA50 > EMA200)
        # 3. Volume confirmation
        if (rsi[i] < 30) and \
           (trend_4h_aligned[i] == 1) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 1h RSI > 70 (overbought)
        # 2. 4h trend = downtrend (EMA50 < EMA200)
        # 3. Volume confirmation
        elif (rsi[i] > 70) and \
             (trend_4h_aligned[i] == -1) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_RSI14_4hEMA50_200_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0