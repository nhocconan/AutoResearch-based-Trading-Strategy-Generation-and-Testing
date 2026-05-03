#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter
# Long when price breaks above Donchian(20) high, ADX(14) > 25, and volume > 1.5x 20-bar average
# Short when price breaks below Donchian(20) low, ADX(14) > 25, and volume > 1.5x 20-bar average
# Uses ADX for trend strength to avoid whipsaws in ranging markets
# Volume spike confirms breakout momentum
# Discrete position sizing (0.25) to minimize fee churn
# Designed for moderate trade frequency (~50-100/year on 4h) to balance opportunity and cost
# Works in bull (breakouts with rising ADX) and bear (breakdowns with rising ADX)

name = "4h_Donchian20_Volume_ADX25_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ADX(14) for trend strength
    # ADX requires +DI, -DI, and TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 14, 14) + 1  # Donchian(20) + ADX(14) + DI(14) + volume MA(20) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, ADX > 25, volume spike
            if (close[i] > donchian_high[i] and 
                adx[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, ADX > 25, volume spike
            elif (close[i] < donchian_low[i] and 
                  adx[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian low or ADX < 20 (trend weakening)
            if (close[i] < donchian_low[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian high or ADX < 20 (trend weakening)
            if (close[i] > donchian_high[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals