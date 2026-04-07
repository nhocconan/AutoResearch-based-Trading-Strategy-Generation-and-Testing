#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour EMA trend filter and 1-day RSI mean reversion
# Uses 4-hour EMA(34) for trend direction and 1-day RSI(14) for overbought/oversold signals
# Designed to work in both bull and bear markets by combining trend filter with mean reversion
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Uses session filter (08-20 UTC) to avoid low-liquidity periods
# Position size: 0.20 (20% of capital) to limit drawdown during adverse moves

name = "1h_ema_trend_rsi_meanrev_4h1d_v1"
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
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1-day data for RSI mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            if position != 0:
                signals[i] = position * 0.20  # maintain position outside session
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI crosses above 70 (overbought)
            if rsi_14_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI crosses below 30 (oversold)
            if rsi_14_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Trend filter: 4h EMA(34) slope determines trend
            if i >= 51:
                ema_now = ema_34_4h_aligned[i]
                ema_prev = ema_34_4h_aligned[i-1]
                uptrend = ema_now > ema_prev
                downtrend = ema_now < ema_prev
            else:
                uptrend = ema_34_4h_aligned[i] > ema_34_4h_aligned[0]
                downtrend = ema_34_4h_aligned[i] < ema_34_4h_aligned[0]
            
            # Mean reversion signals from 1-day RSI
            rsi = rsi_14_1d_aligned[i]
            oversold = rsi < 30
            overbought = rsi > 70
            
            # Long: uptrend + RSI oversold (pullback in uptrend)
            if uptrend and oversold:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + RSI overbought (pullback in downtrend)
            elif downtrend and overbought:
                signals[i] = -0.20
                position = -1
    
    return signals