#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h EMA trend filter with 1d RSI mean reversion entries.
# Long when 4h EMA(50) bullish and 1h RSI(14) < 30 (oversold), short when 4h EMA(50) bearish and 1h RSI > 70 (overbought).
# Uses 1h only for entry timing, 4h for trend direction, 1d RSI for additional overbought/oversold confirmation.
# Time-based filter: trade only 08-20 UTC to avoid low-liquidity hours.
# Target: 20-40 trades/year by requiring trend alignment + RSI extremes + session filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14) for overbought/oversold filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Pre-calculate 1h RSI(14) for entry signals
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-calculate session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 4h EMA slope (bullish if current > previous)
        ema_bullish = ema_4h_aligned[i] > ema_4h_aligned[i-1]
        ema_bearish = ema_4h_aligned[i] < ema_4h_aligned[i-1]
        
        # Entry conditions
        if position == 0:
            # Long: 4h EMA bullish + 1h RSI oversold + 1d RSI not extremely overbought
            if ema_bullish and rsi[i] < 30 and rsi_1d_aligned[i] < 70:
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA bearish + 1h RSI overbought + 1d RSI not extremely oversold
            elif ema_bearish and rsi[i] > 70 and rsi_1d_aligned[i] > 30:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: trend reversal or RSI normalization
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if 4h EMA turns bearish or RSI returns to neutral
                if ema_bearish or rsi[i] > 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if 4h EMA turns bullish or RSI returns to neutral
                if ema_bullish or rsi[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA50_RSI_MeanReversion_Session"
timeframe = "1h"
leverage = 1.0