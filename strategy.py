#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian breakout and volume confirmation
# Choppiness Index (CHOP) identifies market regimes: CHOP > 61.8 = ranging (mean reversion),
# CHOP < 38.2 = trending (trend following). In ranging markets, we fade Donchian breaks;
# in trending markets, we follow breakouts. Volume confirms breakout strength.
# Designed to work in both bull and bear markets by adapting to regime.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "4h_Choppiness_Donchian_Volume_Regime"
timeframe = "4h"
leverage = 1.0

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(high)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = 100 * np.log10(atr * period / (highest_high - lowest_low)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h indicators
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    chop = calculate_chop(high, low, close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        chop_val = chop[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            if chop_val < 38.2:  # Trending regime
                # Enter long: Donchian breakout above + volume + uptrend
                if close[i] > donch_high[i] and vol_conf and close[i] > ema50_1d_val:
                    signals[i] = 0.25
                    position = 1
                # Enter short: Donchian breakdown below + volume + downtrend
                elif close[i] < donch_low[i] and vol_conf and close[i] < ema50_1d_val:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging regime
                # Enter long: fade Donchian breakdown (mean reversion) + volume
                if close[i] < donch_low[i] and vol_conf:
                    signals[i] = 0.20
                    position = 1
                # Enter short: fade Donchian breakout (mean reversion) + volume
                elif close[i] > donch_high[i] and vol_conf:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: opposite Donchian touch OR regime change against position
            if chop_val < 38.2:  # Still trending
                if close[i] < donch_low[i]:  # Exit on breakdown
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging or transitioning
                if close[i] > donch_high[i]:  # Exit on breakout (took profit)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit short: opposite Donchian touch OR regime change against position
            if chop_val < 38.2:  # Still trending
                if close[i] > donch_high[i]:  # Exit on breakout
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging or transitioning
                if close[i] < donch_low[i]:  # Exit on breakdown (took profit)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals