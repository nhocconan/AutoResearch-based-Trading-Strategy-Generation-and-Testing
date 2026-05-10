#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Breakout above/below Camarilla R1/S1 levels on 1h, filtered by 4h EMA50 trend and volume confirmation (>1.5x average).
# Uses 1h ATR-based stoploss. Designed for 15-37 trades/year on 1h to avoid fee drag. Works in bull/bear via 4h trend filter.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Camarilla levels (1h) - calculated from previous day's OHLC
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_range = np.full(n, np.nan)
    
    # Calculate daily OHLC for Camarilla
    df = prices.copy()
    df['date'] = pd.to_datetime(df['open_time']).dt.date
    daily_agg = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    daily_high = daily_agg['high'].values
    daily_low = daily_agg['low'].values
    daily_close = daily_agg['close'].values
    
    # Map daily values to each 1h bar
    date_map = pd.Series(df['date'].values)
    daily_high_map = np.full(n, np.nan)
    daily_low_map = np.full(n, np.nan)
    daily_close_map = np.full(n, np.nan)
    
    for date_val, dh, dl, dc in zip(daily_agg.index, daily_high, daily_low, daily_close):
        mask = (date_map == date_val)
        daily_high_map[mask] = dh
        daily_low_map[mask] = dl
        daily_close_map[mask] = dc
    
    # Camarilla R1 and S1
    camarilla_high = daily_close_map + 1.1 * (daily_high_map - daily_low_map) / 12
    camarilla_low = daily_close_map - 1.1 * (daily_high_map - daily_low_map) / 12
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 4h EMA50 trend
            if close[i] > ema_50_4h_aligned[i]:  # Uptrend
                # Long: Breakout above Camarilla R1 with volume confirmation
                if close[i] > camarilla_high[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.20
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Camarilla S1 with volume confirmation
                if close[i] < camarilla_low[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA50 or stoploss hit
            if close[i] < ema_50_4h_aligned[i] or (i > 0 and low[i] < camarilla_low[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price closes above EMA50 or stoploss hit
            if close[i] > ema_50_4h_aligned[i] or (i > 0 and high[i] > camarilla_high[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals