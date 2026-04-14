#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI mean reversion with 1d trend filter and session filter.
# Long when 4h RSI < 30 AND 1d EMA50 > EMA200 (bullish trend) AND hour in 08-20 UTC.
# Short when 4h RSI > 70 AND 1d EMA50 < EMA200 (bearish trend) AND hour in 08-20 UTC.
# Exit when 4h RSI crosses 50 (mean reversion complete) OR trend weakens (EMA50/EMA200 cross).
# Uses 4h for signal direction (RSI extremes) and 1d for trend filter, 1h only for entry timing.
# Session filter reduces noise trades outside active hours. Target: 15-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute hour filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 4h data ONCE for RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI (14-period)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_rsi(gain, loss, period):
        rsi = np.full_like(close_4h, np.nan)
        if len(gain) < period:
            return rsi
        avg_gain = np.nansum(gain[1:period+1]) / period
        avg_loss = np.nansum(loss[1:period+1]) / period
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rsi[period] = 100 - (100 / (1 + avg_gain / avg_loss))
        for i in range(period+1, len(close_4h)):
            avg_gain = (avg_gain * (period-1) + gain[i]) / period
            avg_loss = (avg_loss * (period-1) + loss[i]) / period
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss))
        return rsi
    
    rsi_4h = wilders_rsi(gain, loss, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Load 1d data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 and EMA200
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).values
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(30, 50, 200)  # Need 4h RSI and 1d EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for mean reversion entries with trend filter
            # Long: 4h RSI oversold (<30) AND bullish trend (EMA50 > EMA200)
            if (rsi_4h_aligned[i] < 30 and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: 4h RSI overbought (>70) AND bearish trend (EMA50 < EMA200)
            elif (rsi_4h_aligned[i] > 70 and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses 50 (mean reversion) OR trend weakens
            if (rsi_4h_aligned[i] > 50 or 
                ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses 50 (mean reversion) OR trend weakens
            if (rsi_4h_aligned[i] < 50 or 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_RSI_MeanReversion_1dTrend_Filter_v1"
timeframe = "1h"
leverage = 1.0