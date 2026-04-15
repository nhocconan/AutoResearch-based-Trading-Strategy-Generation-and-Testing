#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA(34) trend filter and volume confirmation
# Long when RSI < 30 (oversold) + price > 4h EMA34 (uptrend bias) + volume > 1.3x 20-period avg
# Short when RSI > 70 (overbought) + price < 4h EMA34 (downtrend bias) + volume > 1.3x 20-period avg
# Uses 1h timeframe for precise entry timing, 4h EMA for trend direction filter.
# Designed for low trade frequency: ~15-25 trades/year per symbol.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # === 4h Indicator: EMA(34) (trend direction filter) ===
    close_4h = df_4h['close'].values
    ema_span = 34
    ema_4h = pd.Series(close_4h).ewm(span=ema_span, adjust=False, min_periods=ema_span).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1h Indicators: RSI(14) and Volume SMA(20) ===
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    period_rsi = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[period_rsi] = np.mean(gain[1:period_rsi+1])
    avg_loss[period_rsi] = np.mean(loss[1:period_rsi+1])
    
    for i in range(period_rsi+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period_rsi-1) + gain[i]) / period_rsi
        avg_loss[i] = (avg_loss[i-1] * (period_rsi-1) + loss[i]) / period_rsi
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume SMA for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period_rsi+1, 20) + 5  # RSI(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. RSI < 30 (oversold)
        # 2. Price > 4h EMA34 (uptrend bias)
        # 3. Volume confirmation
        if (rsi[i] < 30) and \
           (close[i] > ema_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. RSI > 70 (overbought)
        # 2. Price < 4h EMA34 (downtrend bias)
        # 3. Volume confirmation
        elif (rsi[i] > 70) and \
             (close[i] < ema_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_RSI14_4hEMA34_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0