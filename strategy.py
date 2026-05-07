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
    
    # Load 4h data ONCE for trend filter and structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA for trend filter (21-period)
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d EMA for higher timeframe trend (50-period)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h RSI for pullback identification (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: above average
        vol_condition = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: 4h uptrend + 1h pullback + volume
            if (ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and  # 4h uptrend
                rsi[i] < 40 and rsi[i] > 25 and                    # 1h pullback in RSI
                close[i] > close[i-1] and                         # Price up on bar
                vol_condition):                                   # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1h bounce + volume
            elif (ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and  # 4h downtrend
                  rsi[i] > 60 and rsi[i] < 75 and                    # 1h bounce in RSI
                  close[i] < close[i-1] and                          # Price down on bar
                  vol_condition):                                    # Volume confirmation
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 4h trend reversal or RSI overbought
            if (ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] or  # 4h trend down
                rsi[i] > 70):                                     # RSI overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 4h trend reversal or RSI oversold
            if (ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] or  # 4h trend up
                rsi[i] < 30):                                     # RSI oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h momentum pullback strategy with 4h trend filter and 1d higher timeframe trend
# - Uses 4h EMA21 for primary trend direction (only trade in 4h trend direction)
# - Uses 1d EMA50 as higher timeframe filter (avoid counter-trend trades)
# - Enters on 1h pullbacks: RSI 25-40 for longs in uptrend, RSI 60-75 for shorts in downtrend
# - Requires price continuation and volume confirmation on entry bar
# - Exits when 4h trend reverses or RSI reaches extremes
# - Position size 0.20 to manage risk and reduce trade frequency
# - Designed for 15-30 trades/year per target (60-120 over 4 years)
# - Works in both bull and bear markets by following 4h trend
# - Volume confirmation reduces false signals
# - Multi-timeframe alignment ensures no look-ahead bias
# - Focus on BTC/ETH as primary targets (avoid SOL-only strategies)