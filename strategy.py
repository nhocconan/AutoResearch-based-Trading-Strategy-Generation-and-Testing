#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily volume confirmation and 1-week EMA50 trend filter
# Long when price breaks above 20-period Donchian high, volume > 1.5x daily average, and weekly close > weekly EMA50
# Short when price breaks below 20-period Donchian low, volume > 1.5x daily average, and weekly close < weekly EMA50
# Exit on opposite Donchian break (below Donchian low for long, above Donchian high for short)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Donchian from 12h, volume confirmation from 1d, trend filter from 1w
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_vol_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily average volume (20-day)
    volume_1d = df_1d['volume'].values
    avg_vol_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_vol_20d_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low (20-period)
            elif close[i] < np.min(low[i-20:i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high (20-period)
            elif close[i] > np.max(high[i-20:i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Calculate Donchian channels from 20 periods ago
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
            
            # Volume confirmation: current volume > 1.5x 20-day average
            vol_confirm = volume[i] > 1.5 * avg_vol_20d_aligned[i]
            
            # Trend filter: weekly close above/below EMA50
            uptrend = close[i] > ema_50_1w_aligned[i]
            downtrend = close[i] < ema_50_1w_aligned[i]
            
            # Long: break above Donchian high with volume confirmation and uptrend
            if close[i] > donchian_high and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low with volume confirmation and downtrend
            elif close[i] < donchian_low and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals