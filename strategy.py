#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend-following strategy using 1d trend filter (EMA50) with 1h ADX for trend strength
# and volume confirmation. Only trades in strong trends (ADX > 25) with price above/below 1d EMA50.
# Uses volume spike (1.5x 20-period average) for entry confirmation. Targets 15-30 trades/year
# by combining 1d trend filter + 1h ADX + volume spike. Works in both bull and bear markets
# by following the higher timeframe trend. Uses discrete position sizing (0.20) to minimize churn.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(low, 1)), np.abs(low - np.roll(high, 1))))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1h volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            continue
        
        # Skip if not enough data
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx[i]) or np.isnan(volume_ma[i]):
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Long conditions: price above 1d EMA50, ADX > 25, volume spike
        if close[i] > ema_50_1d_aligned[i] and adx[i] > 25 and vol_confirm and position <= 0:
            position = 1
            signals[i] = position_size
        # Short conditions: price below 1d EMA50, ADX > 25, volume spike
        elif close[i] < ema_50_1d_aligned[i] and adx[i] > 25 and vol_confirm and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit: ADX falls below 20 (weakening trend) or price crosses back over EMA50
        elif position == 1 and (adx[i] < 20 or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx[i] < 20 or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_1d_EMA50_ADX_Volume"
timeframe = "1h"
leverage = 1.0