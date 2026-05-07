#!/usr/bin/env python3
name = "1h_4h1d_Momentum_Pullback_Volume"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI(14) for momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1h volume spike detection: 24-period average (1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback in 4h uptrend with 1d momentum and volume
            uptrend_4h = close[i] > ema_50_4h_aligned[i]
            momentum_up = rsi_14_1d_aligned[i] > 50 and rsi_14_1d_aligned[i] > rsi_14_1d_aligned[i-1]
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            
            if uptrend_4h and momentum_up and vol_condition:
                signals[i] = 0.20
                position = 1
            # Short: pullback in 4h downtrend with 1d momentum and volume
            elif not uptrend_4h and (100 - rsi_14_1d_aligned[i]) > 50 and (100 - rsi_14_1d_aligned[i]) > (100 - rsi_14_1d_aligned[i-1]) and vol_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: trend break or momentum fade
            if close[i] < ema_50_4h_aligned[i] * 0.995 or rsi_14_1d_aligned[i] < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: trend break or momentum fade
            if close[i] > ema_50_4h_aligned[i] * 1.005 or rsi_14_1d_aligned[i] > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h momentum pullback strategy with 4h trend filter and 1d momentum confirmation
# - In 4h uptrend: buy pullbacks when 1d RSI shows rising momentum (>50 and increasing)
# - In 4h downtrend: sell pullbacks when 1d RSI shows bearish momentum (<50 and decreasing)
# - Volume confirmation (1.5x daily average) ensures institutional participation
# - Uses 1h timeframe only for entry timing, direction comes from higher timeframes
# - Tight exit conditions: trend break or RSI fading reduces whipsaws
# - Position size 0.20 limits risk while allowing meaningful returns
# - Designed to work in both bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend)
# - Target: 60-150 total trades over 4 years (15-37/year) to stay within fee limits
# - Avoids overtrading by requiring confluence of trend, momentum, and volume
# - Novel combination: 4h trend + 1d momentum + 1h volume not recently tested in isolation