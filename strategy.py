#!/usr/bin/env python3
# 4h_MultiTimeframe_Adaptive_Strategy_v1
# Hypothesis: Multi-timeframe strategy combining 1d trend direction (EMA34), 4h momentum (RSI), and volume confirmation. 
# Uses adaptive position sizing based on volatility (ATR) and avoids overtrading by requiring confluence of multiple conditions.
# Designed for both bull and bear markets: long in uptrend (price > EMA34) with bullish momentum, short in downtrend (price < EMA34) with bearish momentum.
# Includes volatility-based stop loss and cooldown periods to reduce false signals and manage risk.

name = "4h_MultiTimeframe_Adaptive_Strategy_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    # Volatility filter: ATR(14) for dynamic sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Dynamic position size based on volatility (inverse volatility scaling)
    # Normalize ATR to get volatility factor, then scale position size
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_factor = np.clip(atr_ma / (atr + 1e-10), 0.5, 2.0)  # Inverse volatility: higher volatility = smaller position
    base_size = 0.25
    position_size = base_size * vol_factor
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(50, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Uptrend (price > EMA34) + bullish momentum (RSI > 50) + volume confirmation
            if close[i] > ema_34_1d_aligned[i] and rsi[i] > 50 and volume_confirmed[i]:
                signals[i] = position_size[i]
                position = 1
            # SHORT: Downtrend (price < EMA34) + bearish momentum (RSI < 50) + volume confirmation
            elif close[i] < ema_34_1d_aligned[i] and rsi[i] < 50 and volume_confirmed[i]:
                signals[i] = -position_size[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal (price < EMA34) or bearish momentum (RSI < 40)
            if close[i] < ema_34_1d_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
                cooldown = 4  # 4-bar cooldown after exit
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # EXIT SHORT: Trend reversal (price > EMA34) or bullish momentum (RSI > 60)
            if close[i] > ema_34_1d_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
                cooldown = 4  # 4-bar cooldown after exit
            else:
                signals[i] = -position_size[i]
    
    return signals