#!/usr/bin/env python3
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
    
    # Get weekly data for regime detection
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend regime
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Daily ATR for volatility filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_daily = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily range for volatility regime
    daily_range = high_daily - low_daily
    range_ma = pd.Series(daily_range).ewm(span=20, adjust=False, min_periods=20).mean().values
    range_ma_aligned = align_htf_to_ltf(prices, df_daily, range_ma)
    
    # Volume spike filter
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(atr_daily_aligned[i]) or 
            np.isnan(range_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when weekly price above EMA50 (bull regime)
        # and daily volatility is elevated (range > 1.5x average)
        bull_regime = close[i] > ema50_weekly_aligned[i]
        high_volatility = daily_range[i] > (range_ma_aligned[i] * 1.5)
        
        if not (bull_regime and high_volatility):
            signals[i] = 0.0
            position = 0
            continue
        
        # Long condition: price above weekly EMA50 + volume spike
        if bull_regime and volume_spike[i]:
            signals[i] = 0.25
            position = 1
        # Exit: price drops below weekly EMA50
        elif position == 1 and close[i] < ema50_weekly_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA50_VolumeSpike_BullRegime"
timeframe = "1d"
leverage = 1.0