#!/usr/bin/env python3
# 1d_Weekly_HTF_Pullback_LongOnly
# Hypothesis: On daily timeframe, pullbacks to the 20-period EMA during a weekly uptrend
# (price above 20-week EMA) with volume confirmation capture institutional accumulation.
# In bull markets, this captures trend continuation; in bear markets, avoids shorts and
# only takes longs during rare bullish pullbacks, reducing whipsaw. Volume filter reduces
# false signals. Target: 10-25 trades per year (~40-100 over 4 years) with position size 0.25.

name = "1d_Weekly_HTF_Pullback_LongOnly"
timeframe = "1d"
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
    
    # Load weekly data ONCE for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter (uptrend when price > weekly EMA20)
    ema_20_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily EMA20 for pullback entry
    ema_20_daily = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume ratio: current volume / 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 20  # Need 20 periods for daily EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_weekly_aligned[i]) or np.isnan(ema_20_daily[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Pullback condition: price touches or slightly below daily EMA20
        pullback = close[i] <= ema_20_daily[i] * 1.005  # Allow 0.5% above EMA for noise
        
        # Weekly uptrend filter: price above weekly EMA20
        weekly_uptrend = close[i] > ema_20_weekly_aligned[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Enter long on pullback during weekly uptrend with volume
            if pullback and weekly_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit: price rises significantly above EMA20 (take profit) or weekly trend breaks
            if close[i] > ema_20_daily[i] * 1.08 or not weekly_uptrend:  # 8% profit target or trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals