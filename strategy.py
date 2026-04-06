#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and session filter (08-20 UTC)
# Uses RSI(14) extreme values (<30 for long, >70 for short) but only in direction of higher timeframe trend
# 4h EMA(50) determines trend: long only when price > EMA50, short only when price < EMA50
# 1d volume filter: requires volume > 1.5x 20-period average to confirm strength
# Session filter: only trade 08:00-20:00 UTC to avoid low-volume Asian session
# Target: 60-150 trades over 4 years by combining multiple filters for precision

name = "1h_rsi_meanrev_4h1d_trend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_threshold_1d = 1.5 * volume_ma_20
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    volume_threshold_aligned = align_htf_to_ltf(prices, df_1d, volume_threshold_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or np.isnan(volume_threshold_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI returns to neutral (50) or breaks above 70
            if rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (50) or breaks below 30
            if rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + trend filter + volume confirmation
            if volume_1d_aligned[i] > volume_threshold_aligned[i]:
                if rsi[i] < 30 and close[i] > ema_50_4h_aligned[i]:
                    # Oversold in uptrend: long signal
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] > 70 and close[i] < ema_50_4h_aligned[i]:
                    # Overbought in downtrend: short signal
                    signals[i] = -0.20
                    position = -1
    
    return signals