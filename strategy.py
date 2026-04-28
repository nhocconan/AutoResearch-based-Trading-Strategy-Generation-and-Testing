# 4H_RSI_Momentum_Breaker
# Hypothesis: 4H RSI momentum combined with volume breakout and ADX trend filter captures strong momentum moves while avoiding false signals in ranging markets.
# Works in bull/bear: RSI momentum identifies strong moves, volume confirms institutional participation, ADX filters out chop.
# Target: 20-40 trades/year on 4h timeframe to minimize fee drag.
# Uses 4h RSI(14) with momentum, volume spike, and ADX(14) > 25 for trend strength.

#!/usr/bin/env python3
import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate RSI momentum (rate of change)
    rsi_momentum = np.diff(rsi, prepend=rsi[0])
    
    # Calculate volume spike (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # Calculate ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(rsi_momentum[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        # Volume filter: volume spike > 1.5x average
        vol_spike = vol_ratio[i] > 1.5
        
        # RSI momentum conditions
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        rsi_momentum_up = rsi_momentum[i] > 5
        rsi_momentum_down = rsi_momentum[i] < -5
        
        # Entry conditions
        long_entry = rsi_oversold and rsi_momentum_up and vol_spike and strong_trend
        short_entry = rsi_overbought and rsi_momentum_down and vol_spike and strong_trend
        
        # Exit conditions: RSI returns to neutral range or momentum fades
        long_exit = rsi[i] > 50 or rsi_momentum[i] < 0
        short_exit = rsi[i] < 50 or rsi_momentum[i] > 0
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4H_RSI_Momentum_Breaker"
timeframe = "4h"
leverage = 1.0