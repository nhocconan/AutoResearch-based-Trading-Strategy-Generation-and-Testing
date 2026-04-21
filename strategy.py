#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily High-Low Range as volatility proxy ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range (high - low)
    daily_range = high_1d - low_1d
    
    # 20-day average range
    avg_range_20 = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    # Current range ratio to 20-day average
    range_ratio = daily_range / avg_range_20
    
    # Range percentile (252-day lookback for regime)
    range_percentile = pd.Series(range_ratio).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align range percentile to 4h timeframe
    range_percentile_aligned = align_htf_to_ltf(prices, df_1d, range_percentile)
    
    # === Daily EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(range_percentile_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        range_pct = range_percentile_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + uptrend + volume
            if (range_pct < 30 and  # Low volatility regime
                price_close > ema_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (range) + downtrend + volume
            elif (range_pct < 30 and   # Low volatility regime
                  price_close < ema_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or opposite condition
            if position == 1 and (range_pct > 70 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (range_pct > 70 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyRange_Percentile_EMA34_Volume"
timeframe = "4h"
leverage = 1.0