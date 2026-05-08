#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_RSI_Pullback_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily RSI(14) for pullback entries
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smooth(gain, 14)
    avg_loss = wilders_smooth(loss, 14)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly EMA50 slope
        if i >= start_idx + 1:
            ema_prev = ema50_1w_aligned[i-1]
            ema_curr = ema50_1w_aligned[i]
            if not (np.isnan(ema_prev) or np.isnan(ema_curr)):
                trend_up = ema_curr > ema_prev
                trend_down = ema_curr < ema_prev
            else:
                trend_up = trend_down = False
        else:
            trend_up = trend_down = False
        
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: uptrend + RSI pullback (< 40)
            if trend_up and rsi_val < 40:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI bounce (> 60)
            elif trend_down and rsi_val > 60:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend reversal or RSI overbought
            if not trend_up or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend reversal or RSI oversold
            if not trend_down or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily RSI pullbacks in the direction of weekly trend capture
# intermediate swings in both bull and bear markets. Weekly EMA50 filter
# ensures we only trade with the higher timeframe trend, reducing whipsaw.
# RSI < 40 for longs and > 60 for shorts provides oversold/overbought
# entries during pullbacks. Exits on trend reversal or RSI extremes.
# Weekly timeframe reduces noise, daily provides timely entries.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.