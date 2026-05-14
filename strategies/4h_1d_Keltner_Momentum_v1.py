#!/usr/bin/env python3
"""
4h_1d_Keltner_Momentum_v1
Hypothesis: Trade breakouts from Keltner Channel (EMA20 ± 2*ATR(10)) with momentum confirmation (RSI > 55 for long, < 45 for short) and volume > 1.3x average. 
Use 1-day ADX > 20 to ensure trending market. Designed for 25-40 trades/year with clear trend-following logic that works in bull (breakouts continue) and bear (breakouts fail, reverse) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Keltner_Momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR ADX TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4H INDICATORS: KELTNER CHANNEL AND MOMENTUM ===
    # EMA20 for Keltner middle
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for Keltner width
    atr_period = 10
    tr_4h = np.zeros_like(high)
    for i in range(1, len(high)):
        tr_4h[i] = max(high[i] - low[i], 
                      abs(high[i] - close[i-1]), 
                      abs(low[i] - close[i-1]))
    atr_10 = pd.Series(tr_4h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Keltner Bands
    kc_upper = ema20 + 2 * atr_10
    kc_lower = ema20 - 2 * atr_10
    
    # RSI(14) for momentum
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.3)
        
        # Long: price breaks above Keltner Upper with momentum and volume
        long_signal = (close[i] > kc_upper[i] and 
                      rsi[i] > 55 and 
                      strong_volume and 
                      trending)
        
        # Short: price breaks below Keltner Lower with momentum and volume
        short_signal = (close[i] < kc_lower[i] and 
                       rsi[i] < 45 and 
                       strong_volume and 
                       trending)
        
        # Exit: price returns to middle (EMA20) or trend weakens
        exit_long = (position == 1 and 
                    (close[i] < ema20[i] or adx_aligned[i] < 15))
        exit_short = (position == -1 and 
                     (close[i] > ema20[i] or adx_aligned[i] < 15))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals