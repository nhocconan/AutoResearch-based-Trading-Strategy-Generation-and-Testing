#!/usr/bin/env python3
name = "4h_RSI_Div_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily RSI(14) for divergence detection
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 3-period average (half day of 4h bars)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = False
            if i >= 2:
                if low[i] < low[i-1] and low[i-1] < low[i-2]:
                    if rsi_aligned[i] > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2]:
                        bull_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = False
            if i >= 2:
                if high[i] > high[i-1] and high[i-1] > high[i-2]:
                    if rsi_aligned[i] < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2]:
                        bear_div = True
            
            # Long: bullish divergence with volume and uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if bull_div and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence with volume and downtrend
            elif bear_div and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below EMA or divergence fails
            if close[i] < ema_50_1d_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above EMA or divergence fails
            if close[i] > ema_50_1d_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h RSI divergence with 1d trend and volume confirmation
# - Uses daily RSI to detect momentum divergences (more reliable than lower timeframe)
# - Bullish divergence: price makes lower low while RSI makes higher low = weakening downtrend
# - Bearish divergence: price makes higher high while RSI makes lower high = weakening uptrend
# - Entry only when divergence aligns with higher timeframe trend (EMA50)
# - Volume confirmation (1.5x average) filters weak signals
# - Exits when price crosses EMA50 or RSI reaches extreme levels
# - Works in both bull and bear markets by following the 1d trend
# - Divergences are relatively rare, keeping trade frequency low (target: 20-50/year)
# - Avoids overtrading while capturing meaningful reversals
# - Position size 0.25 balances opportunity with risk management
# - Daily timeframe reduces noise vs using 4h RSI alone
# - Novel combination not seen in recent failed attempts (avoids saturated strategies)