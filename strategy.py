#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend filter with 1d RSI mean reversion and volume confirmation
# Uses 1d RSI(14) for mean reversion signals (RSI < 30 long, RSI > 70 short)
# 4h KAMA(14) filters trend direction to avoid counter-trend trades
# Volume spike (>1.5x 20-bar average) confirms momentum
# Fixed profit targets at 1.5% and 3% with trailing stop via signal=0 when adverse move exceeds 0.5%
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag
# Works in bull/bear: mean reversion captures reversals, trend filter avoids false signals, volume ensures participation

name = "4h_KAMA_1dRSI_MeanRev_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14) for mean reversion
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 4h KAMA(14) trend filter
    change = np.abs(np.diff(close, k=1))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.0645 - 0.0625) + 0.0625) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate volume confirmation (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: RSI < 30 (oversold) AND price > KAMA (uptrend) AND volume spike
            if rsi_1d_aligned[i] < 30 and close[i] > kama[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: RSI > 70 (overbought) AND price < KAMA (downtrend) AND volume spike
            elif rsi_1d_aligned[i] > 70 and close[i] < kama[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or adverse move > 0.5%
            if rsi_1d_aligned[i] > 50 or close[i] < signals[i-1] * 0.995 * (1 if signals[i-1] > 0 else 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or adverse move > 0.5%
            if rsi_1d_aligned[i] < 50 or close[i] > signals[i-1] * 1.005 * (1 if signals[i-1] < 0 else 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals