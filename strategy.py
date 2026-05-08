#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d ADX filter + volume confirmation
# Long when: Alligator bullish (JAW > TEETH > LIPS), ADX > 25 (trending), volume > 1.5x average
# Short when: Alligator bearish (LIPS < TEETH < JAW), ADX > 25 (trending), volume > 1.5x average
# Uses Alligator for trend direction and ADX to filter ranging markets.
# Volume spike confirms institutional participation.
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.

name = "6h_WilliamsAlligator_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data once for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up = df_1d['high'].diff()
    down = df_1d['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 6h
    # SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values  # Blue line (13-period)
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values    # Red line (8-period)
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values    # Green line (5-period)
    
    # Shift jaws, teeth, lips forward as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values become 0 after roll
    jaw[:8] = 0
    teeth[:5] = 0
    lips[:3] = 0
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator signals
        bullish = jaw[i] > teeth[i] > lips[i]
        bearish = lips[i] < teeth[i] < jaw[i]
        
        # ADX filter: trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Enter long: Alligator bullish + trending + volume spike
            if bullish and trending and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish + trending + volume spike
            elif bearish and trending and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish or loses trend
            if not bullish or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish or loses trend
            if not bearish or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals