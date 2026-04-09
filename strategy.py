#!/usr/bin/env python3
# 1h_rsi_pullback_4h1d_trend_v1
# Hypothesis: 1h RSI pullback strategy with 4h/1d HTF trend alignment for BTC/ETH/SOL.
# Long: RSI(14) < 30 (oversold) + price > 4h EMA(50) + price > 1d EMA(200) (bullish alignment)
# Short: RSI(14) > 70 (overbought) + price < 4h EMA(50) + price < 1d EMA(200) (bearish alignment)
# Exit: RSI crosses back to neutral zone (40-60) or opposite extreme RSI reached
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Position size: 0.20 discrete levels to minimize fee churn.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 4h EMA(50) for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    ema_50_4h = close_4h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(200) for long-term trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_200_1d = close_1d_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup for 1d EMA(200)
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 40 (exit oversold) OR RSI > 70 (overbought reversal)
            if rsi_values[i] > 40 or rsi_values[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 60 (exit overbought) OR RSI < 30 (oversold reversal)
            if rsi_values[i] < 60 or rsi_values[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long setup: RSI oversold + bullish HTF alignment
            long_setup = (rsi_values[i] < 30) and (close[i] > ema_50_4h_aligned[i]) and (close[i] > ema_200_1d_aligned[i])
            # Short setup: RSI overbought + bearish HTF alignment
            short_setup = (rsi_values[i] > 70) and (close[i] < ema_50_4h_aligned[i]) and (close[i] < ema_200_1d_aligned[i])
            
            if long_setup:
                position = 1
                signals[i] = 0.20
            elif short_setup:
                position = -1
                signals[i] = -0.20
    
    return signals