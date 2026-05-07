#!/usr/bin/env python3
name = "6h_Keltner_RSI_MeanRev_TrendFilter"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Keltner Channel (20, 2.0) on 6h
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_mid + 2 * atr
    lower_keltner = ema_mid - 2 * atr
    
    # RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_mid[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(rsi[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below lower Keltner + RSI oversold + daily uptrend
            if close[i] < lower_keltner[i] and rsi[i] < 30 and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price above upper Keltner + RSI overbought + daily downtrend
            elif close[i] > upper_keltner[i] and rsi[i] > 70 and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back to middle Keltner or RSI neutral
            if close[i] > ema_mid[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back to middle Keltner or RSI neutral
            if close[i] < ema_mid[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Keltner Channel mean reversion with RSI and daily trend filter
# - Mean reversion at Keltner bands (2*ATR from EMA20) in ranging markets
# - RSI confirms overbought/oversold conditions (RSI<30 for long, >70 for short)
# - Daily EMA34 trend filter ensures we only trade mean reversion in the direction of higher timeframe trend
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Exit when price returns to EMA20 or RSI normalizes
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Combines proven mean reversion with trend filter for robustness across regimes