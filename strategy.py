#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly EMA25/EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper(20) in uptrend with volume > 2.0x average
# Short when price breaks below Donchian lower(20) in downtrend with volume > 2.0x average
# Exit on opposite Donchian break or 2.5*ATR stoploss
# Position size: 0.30 (30% of capital)
# Target: ~15-30 trades/year on 12h (60-120 total over 4 years) to minimize fee drag

name = "12h_donchian20_weekly_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA(25) and EMA(50) for trend filter
    close_weekly = df_weekly['close'].values
    ema_25_weekly = pd.Series(close_weekly).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_25_weekly)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (30-period)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(ema_25_weekly_aligned[i]) or np.isnan(ema_50_weekly_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian lower(20)
            elif close[i] < low[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian upper(20)
            elif close[i] > high[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Calculate Donchian channels (20-period)
            highest_high = high[i-20:i].max() if i >= 20 else high[:i].max()
            lowest_low = low[i-20:i].min() if i >= 20 else low[:i].min()
            
            # Trend filter: weekly EMA(25) > EMA(50) for uptrend, < for downtrend
            uptrend = ema_25_weekly_aligned[i] > ema_50_weekly_aligned[i]
            downtrend = ema_25_weekly_aligned[i] < ema_50_weekly_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * average volume
            volume_confirm = volume[i] > 2.0 * vol_avg[i]
            
            # Long: price breaks above Donchian upper(20) in uptrend with volume
            if close[i] > highest_high and uptrend and volume_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower(20) in downtrend with volume
            elif close[i] < lowest_low and downtrend and volume_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals