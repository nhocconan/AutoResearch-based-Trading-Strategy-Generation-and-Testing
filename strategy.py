#!/usr/bin/env python3
# 1d_1w_Momentum_MeanReversion_v1
# Hypothesis: On daily timeframe, use 1-week trend filter (price above/below weekly EMA50) combined with daily RSI mean reversion (RSI<30 for long, RSI>70 for short) and volume confirmation (1.5x 20-day average). Designed for low trade frequency (<15/year) to avoid fee drag. Works in bull/bear via weekly trend filter and mean reversion logic.

name = "1d_1w_Momentum_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: RSI oversold in bullish weekly trend with volume surge
            if rsi[i] < 30 and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in bearish weekly trend with volume surge
            elif rsi[i] > 70 and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long when RSI returns to neutral (>=50)
                if rsi[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when RSI returns to neutral (<=50)
                if rsi[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals