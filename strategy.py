#!/usr/bin/env python3
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
    
    # Get weekly HTF data once before loop (6h primary, 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate weekly pivot points (based on previous week)
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    prev_weekly_close = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    prev_weekly_high = np.concatenate([[weekly_high[0]], weekly_high[:-1]])
    prev_weekly_low = np.concatenate([[weekly_low[0]], weekly_low[:-1]])
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_range = prev_weekly_high - prev_weekly_low
    weekly_r1 = 2 * weekly_pivot - prev_weekly_low
    weekly_s1 = 2 * weekly_pivot - prev_weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # Calculate weekly EMA20 for trend filter
    weekly_ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly RSI(14) for momentum filter
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    weekly_rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 6h timeframe with proper delay
    weekly_ema_20_6h = align_htf_to_ltf(prices, df_1w, weekly_ema_20)
    weekly_rsi_14_6h = align_htf_to_ltf(prices, df_1w, weekly_rsi_14)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_20_6h[i]) or np.isnan(weekly_rsi_14_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or
            np.isnan(weekly_r2_6h[i]) or np.isnan(weekly_s2_6h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA20
        # 2. Weekly momentum filter: RSI not extreme (avoid overbought/oversold)
        # 3. 6h Donchian breakout: price breaks 20-period high/low
        # 4. Weekly pivot confirmation: price near R1/S1 for continuation
        # 5. Volume confirmation: volume > 1.3x average
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend, near weekly R1
        if (close[i] > weekly_ema_20_6h[i] and          # Weekly uptrend filter
            weekly_rsi_14_6h[i] < 70 and                # Not overbought
            close[i] > highest_20[i] and                # Donchian breakout
            close[i] >= weekly_r1_6h[i] * 0.995 and     # Near weekly R1 (within 0.5%)
            volume_ratio[i] > 1.3):                     # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend, near weekly S1
        elif (close[i] < weekly_ema_20_6h[i] and       # Weekly downtrend filter
              weekly_rsi_14_6h[i] > 30 and              # Not oversold
              close[i] < lowest_20[i] and               # Donchian breakdown
              close[i] <= weekly_s1_6h[i] * 1.005 and   # Near weekly S1 (within 0.5%)
              volume_ratio[i] > 1.3):                   # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Donchian_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0