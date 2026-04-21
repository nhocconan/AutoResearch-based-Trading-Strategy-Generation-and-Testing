#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h RSI(14) mean reversion and 1d trend filter.
# Long when 4h RSI < 30 (oversold) and 1d EMA50 > EMA200 (bullish trend).
# Short when 4h RSI > 70 (overbought) and 1d EMA50 < EMA200 (bearish trend).
# Exit when RSI returns to neutral (40-60) or trend weakens.
# Uses 4h for signal direction (mean reversion extremes) and 1d for trend filter.
# 1h only for entry timing precision. Target: 20-40 trades/year.
# Works in bull/bear: trend filter ensures we only trade mean reversion in direction of higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
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
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if data not ready
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
        bullish_trend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        bearish_trend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: 4h RSI oversold (<30) in bullish trend
            if rsi_4h_aligned[i] < 30 and bullish_trend:
                signals[i] = 0.20
                position = 1
            # Short: 4h RSI overbought (>70) in bearish trend
            elif rsi_4h_aligned[i] > 70 and bearish_trend:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if RSI returns to neutral (>=40) or trend turns bearish
                if rsi_4h_aligned[i] >= 40 or not bullish_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if RSI returns to neutral (<=60) or trend turns bullish
                if rsi_4h_aligned[i] <= 60 or not bearish_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI14_MeanReversion_4hOBOS_1dEMA50_200_Trend"
timeframe = "1h"
leverage = 1.0