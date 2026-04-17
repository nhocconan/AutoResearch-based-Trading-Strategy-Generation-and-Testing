# 4h_Range_Breakout_With_Trend_Filter
# Hypothesis: In 4h timeframe, enter long when price breaks above Donchian high (20-period) with volume confirmation and price > 200-period EMA; enter short when price breaks below Donchian low with volume confirmation and price < 200 EMA. Use 12h ADX to filter for trending markets only. Designed for low trade frequency to minimize fee drag while capturing strong trending moves in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 200-period EMA for trend filter ===
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # === Volume confirmation (current volume > 1.5x 20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h ADX for trend strength filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = -np.diff(low_12h, prepend=low_12h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200  # For EMA200 and ADX
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(ema200[i]) or
            np.isnan(vol_ma20[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma20[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + volume filter + trend filter + price > EMA200
            if close[i] > donch_high[i] and vol_filter and trend_filter and close[i] > ema200[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volume filter + trend filter + price < EMA200
            elif close[i] < donch_low[i] and vol_filter and trend_filter and close[i] < ema200[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price closes below Donchian low (reversal signal)
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes above Donchian high (reversal signal)
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Range_Breakout_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0