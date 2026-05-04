#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1w Supertrend filter and volume spike confirmation
# KAMA adapts to market noise, reducing whipsaws in ranging markets. Supertrend on 1w
# provides strong trend filter to avoid counter-trend trades. Volume spike ensures
# institutional participation. Designed for low trade frequency (12-37/year) with
# discrete sizing (0.25) to minimize fee drag. Works in bull (trend follow) and bear
# (avoid false signals via Supertrend) markets.

name = "12h_KAMA_1wSupertrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Supertrend (ATR=10, mult=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(df_1w['high']) - df_1w['low']
    tr2 = abs(pd.Series(df_1w['high']) - df_1w['close'].shift(1))
    tr3 = abs(pd.Series(df_1w['low']) - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    hl2 = (df_1w['high'] + df_1w['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(len(df_1w), np.nan)
    direction = np.full(len(df_1w), 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, len(df_1w)):
        # Upper Band
        if i == atr_period:
            upper_band[i] = hl2.iloc[i] + (multiplier * atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan
            lower_band[i] = hl2.iloc[i] - (multiplier * atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan
        else:
            upper_band[i] = hl2.iloc[i] + (multiplier * atr.iloc[i]) if not np.isnan(atr.iloc[i]) else upper_band[i-1]
            lower_band[i] = hl2.iloc[i] - (multiplier * atr.iloc[i]) if not np.isnan(atr.iloc[i]) else lower_band[i-1]
            
            # Adjust bands
            if supertrend[i-1] <= upper_band[i-1]:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            if supertrend[i-1] >= lower_band[i-1]:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        # Trend direction
        if i == atr_period:
            direction[i] = 1 if close.iloc[i] > upper_band[i] else -1
        else:
            if direction[i-1] == -1 and close.iloc[i] > upper_band[i]:
                direction[i] = 1
            elif direction[i-1] == 1 and close.iloc[i] < lower_band[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
        
        # Supertrend value
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    
    # Align 1w Supertrend and direction to 12h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate KAMA on 12h timeframe
    # Efficiency Ratio (ER) over 10 periods
    change = abs(pd.Series(close).diff(10))
    volatility = pd.Series(close).diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing Constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start with first close
    for i in range(10, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Enter long: price above KAMA, 1w uptrend, volume confirmation
            if (close[i] > kama[i] and 
                direction_aligned[i] == 1 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, 1w downtrend, volume confirmation
            elif (close[i] < kama[i] and 
                  direction_aligned[i] == -1 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA OR 1w trend turns down OR volume drops
            if (close[i] < kama[i] or 
                direction_aligned[i] == -1 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA OR 1w trend turns up OR volume drops
            if (close[i] > kama[i] or 
                direction_aligned[i] == 1 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals