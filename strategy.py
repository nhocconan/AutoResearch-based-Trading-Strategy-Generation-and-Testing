#!/usr/bin/env python3
"""
1d Weekly Donchian Channel Breakout with Volume Confirmation and ATR Stop
- Trend: Weekly Donchian channel (20-bar) sets direction
- Entry: Break above/below daily Donchian (20) with volume > 1.5x 20-day average
- Exit: Close crosses 20-day EMA or ATR-based trailing stop
- Designed for low-frequency, high-conviction trades to minimize fee drag
- Works in bull/bear by following weekly trend
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (Donchian 20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Daily Donchian channels (20-period) for entry
    donchian_high_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily 20-period EMA for exit
    ema_20d = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily ATR(14) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all weekly indicators to daily
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after all indicators are ready
    start_idx = max(20, 20, 20, 14)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(donchian_high_20d[i]) or np.isnan(donchian_low_20d[i]) or 
            np.isnan(ema_20d[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above/below weekly Donchian
        weekly_trend = 1 if close[i] > donchian_high_20_aligned[i] else (-1 if close[i] < donchian_low_20_aligned[i] else 0)
        
        # Volume confirmation: current volume > 1.5x weekly average
        vol_confirm = volume[i] > 1.5 * vol_ma_20_1w_aligned[i]
        
        if position == 0:
            # Long: daily break above Donchian high + weekly uptrend + volume
            if close[i] > donchian_high_20d[i] and weekly_trend == 1 and vol_confirm:
                signals[i] = size
                position = 1
            # Short: daily break below Donchian low + weekly downtrend + volume
            elif close[i] < donchian_low_20d[i] and weekly_trend == -1 and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions: close below 20-day EMA OR ATR trailing stop
            # Track highest high since entry for trailing stop
            if i == start_idx or position != 1:  # reset tracking when position changes
                entry_high = high[i]
            else:
                entry_high = max(entry_high, high[i])
            
            exit_condition = (close[i] < ema_20d[i]) or (close[i] < entry_high - 2.5 * atr_14[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: close above 20-day EMA OR ATR trailing stop
            # Track lowest low since entry for trailing stop
            if i == start_idx or position != -1:  # reset tracking when position changes
                entry_low = low[i]
            else:
                entry_low = min(entry_low, low[i])
            
            exit_condition = (close[i] > ema_20d[i]) or (close[i] > entry_low + 2.5 * atr_14[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchianTrend_DailyBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0