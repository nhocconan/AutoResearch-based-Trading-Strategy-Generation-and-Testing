#!/usr/bin/env python3
# 4h_RSI_Trend_Pullback
# Hypothesis: 4-hour RSI pullback in strong trends using 1d EMA trend filter and volume confirmation
# Works in bull markets via long pullbacks to EMA in uptrends, and bear markets via short pullbacks in downtrends
# Volume filter ensures momentum, RSI(14)<40 for long entries and >60 for short entries avoids chasing
# Target: 25-40 trades per year (~100-160 over 4 years) with position size 0.25

name = "4h_RSI_Trend_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need 14 for RSI + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: RSI pullback (<40) in uptrend with volume
            if rsi[i] < 40 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI bounce (>60) in downtrend with volume
            elif rsi[i] > 60 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought (>70) or trend reversal
            if rsi[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold (<30) or trend reversal
            if rsi[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals