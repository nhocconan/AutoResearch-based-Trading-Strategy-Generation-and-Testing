#!/usr/bin/env python3
name = "4h_RSI_Reversion_With_Volume_And_12h_Trend"
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
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA34 on 12h close for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # RSI and volume MA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: RSI oversold (<30) with volume spike in 12h uptrend
            if rsi[i] < 30 and vol_condition and ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with volume spike in 12h downtrend
            elif rsi[i] > 70 and vol_condition and ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral (50) or trend changes
            if rsi[i] >= 50 or ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral (50) or trend changes
            if rsi[i] <= 50 or ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI mean reversion on 4h with volume confirmation and 12h trend filter
# - Long when RSI < 30 (oversold) with 2x volume spike in 12h uptrend
# - Short when RSI > 70 (overbought) with 2x volume spike in 12h downtrend
# - Exit when RSI returns to neutral (50) or 12h trend changes
# - Volume spike filters out weak moves, trend filter avoids counter-trend trades
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - RSI(14) is a proven mean-reversion tool, especially with volume confirmation
# - 12h EMA34 trend filter ensures alignment with higher timeframe momentum
# - Aims for 80-200 total trades over 4 years (20-50/year) within limits