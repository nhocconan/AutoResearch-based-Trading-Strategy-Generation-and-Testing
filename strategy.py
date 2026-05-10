#!/usr/bin/env python3
# 4h_Keltner_Channel_MR_Bounce_1dATR_Filter
# Hypothesis: Mean-reversion bounce off Keltner Channel lower/upper bands on 4h with
# daily ATR filter to avoid ranging markets. Long when price touches lower band with
# RSI < 30 and daily ATR > 20-period average (volatility filter). Short when price
# touches upper band with RSI > 70 and same volatility filter. Exits on middle line
# cross. Designed for 20-35 trades/year to avoid overtrading and work in both
# bull and bear markets by fading extremes in volatile conditions.

name = "4h_Keltner_Channel_MR_Bounce_1dATR_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR(20) on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_20_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ma_20_1d = pd.Series(atr_20_1d).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr_20_1d > atr_ma_20_1d  # Volatility filter: only trade when ATR above average
    
    # Align ATR filter to 4h timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter.astype(float))
    
    # Keltner Channel (20, 2.0) on 4h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_4h = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2.0 * atr_4h
    lower_keltner = ema_20 - 2.0 * atr_4h
    middle_keltner = ema_20  # Exit line
    
    # RSI(14) for overbought/oversold
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Warmup for Keltner and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or np.isnan(rsi[i]) or np.isnan(atr_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Touch lower Keltner band with oversold RSI and volatility filter
            if low[i] <= lower_keltner[i] and rsi[i] < 30 and atr_filter_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Touch upper Keltner band with overbought RSI and volatility filter
            elif high[i] >= upper_keltner[i] and rsi[i] > 70 and atr_filter_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses above middle Keltner line (EMA)
            if close[i] >= middle_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses below middle Keltner line (EMA)
            if close[i] <= middle_keltner[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals