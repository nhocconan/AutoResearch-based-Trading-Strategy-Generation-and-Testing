#!/usr/bin/env python3
name = "6h_ADX_PlusDM_WeeklyTrend_Filter"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly 50-period SMA for trend filter
    weekly_close = df_weekly['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_trend = align_htf_to_ltf(prices, df_weekly, weekly_sma50)
    
    # Calculate ADX and directional movement on 6h data
    period = 14
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # positive values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first element
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values / atr_safe
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    # Volume confirmation: current volume vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(adx[i]) or 
            np.isnan(plus_di[i]) or 
            np.isnan(minus_di[i]) or 
            np.isnan(weekly_trend[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Strong trend filter: weekly price above weekly SMA50
        weekly_uptrend = close[i] > weekly_trend[i]
        weekly_downtrend = close[i] < weekly_trend[i]
        
        if position == 0:
            # Long: ADX > 25 (strong trend), +DI > -DI, weekly uptrend, volume confirmation
            if (adx[i] > 25 and 
                plus_di[i] > minus_di[i] and 
                weekly_uptrend and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend), -DI > +DI, weekly downtrend, volume confirmation
            elif (adx[i] > 25 and 
                  minus_di[i] > plus_di[i] and 
                  weekly_downtrend and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakens (+DI < -DI) or weekly trend changes
            if (plus_di[i] < minus_di[i] or 
                not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakens (-DI < +DI) or weekly trend changes
            if (minus_di[i] < plus_di[i] or 
                not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals