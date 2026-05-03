#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX25 regime filter and volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period volume MA.
# Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period volume MA.
# Elder Ray measures bull/bear power relative to EMA13, ADX filters for trending markets only.
# Works in both bull and bear markets by only trading when trend is strong (ADX>25) and volume confirms.
# Target: 50-150 total trades over 4 years = 12-37/year. Position size: 0.25.

name = "6h_ElderRay_1dADX25_VolumeSpike"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for trend strength
    # ADX = smoothed DX, DX = |DI+ - DI-| / (DI+ + DI-)
    # DI+ = smoothed +DM / ATR, DI- = smoothed -DM / ATR
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # ATR = smoothed TR, TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    
    # Calculate components
    prev_high = np.roll(df_1d['high'], 1)
    prev_low = np.roll(df_1d['low'], 1)
    prev_close = np.roll(df_1d['close'], 1)
    prev_high[0] = df_1d['high'].iloc[0]
    prev_low[0] = df_1d['low'].iloc[0]
    prev_close[0] = df_1d['close'].iloc[0]
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close)
    tr3 = np.abs(df_1d['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = df_1d['high'] - prev_high
    down_move = prev_low - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed ATR, DI+, DI- (using Wilder's smoothing: alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 6h data for Elder Ray and volume
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 6h volume > 1.5x 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Conditions
        strong_trend = adx_aligned[i] > 25
        vol_confirm = volume_spike[i]
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0  # Bear power negative means bears weak
        
        if position == 0:
            # Long: Bull power positive, Bear power negative (bears weak), strong trend, volume spike
            if bull_strong and bear_strong and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative, Bull power positive (bulls weak), strong trend, volume spike
            elif not bull_strong and not bear_strong and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakens OR power shifts
            if not strong_trend or not bull_strong or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakens OR power shifts
            if not strong_trend or not bear_strong or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals