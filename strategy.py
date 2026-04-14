#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX + volume + 1d trend filter
# Uses ADX(14) > 25 to identify trending markets on 12h
# Volume > 1.5x 20-period EMA confirms momentum
# 1d EMA50 filters counter-trend trades (long only above EMA50, short only below)
# Designed for 15-30 trades/year with clear entry/exit rules
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ADX calculation (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):  # Wait for ADX to stabilize
        # Get aligned 1d EMA50
        ema_50_i = align_htf_to_ltf(prices, df_1d, ema_50_1d)[i]
        
        if np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(ema_50_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: ADX trending (+DI > -DI) + price above 1d EMA50 + volume
        if position == 0 and adx[i] > 25 and plus_di[i] > minus_di[i] and close[i] > ema_50_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: ADX trending (-DI > +DI) + price below 1d EMA50 + volume
        elif position == 0 and adx[i] > 25 and minus_di[i] > plus_di[i] and close[i] < ema_50_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Trend weakens (ADX < 20) or opposite DI crossover
        elif position != 0:
            if position == 1 and (adx[i] < 20 or minus_di[i] > plus_di[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (adx[i] < 20 or plus_di[i] > minus_di[i]):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_ADX_Volume_1dEMA50"
timeframe = "12h"
leverage = 1.0