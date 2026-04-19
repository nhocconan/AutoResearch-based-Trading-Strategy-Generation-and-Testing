#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with daily EMA34 trend and volume spike.
# Long when: Choppiness Index > 61.8 (range), price above EMA34, volume > 1.5x 20-period average
# Short when: Choppiness Index < 38.2 (trend), price below EMA34, volume > 1.5x 20-period average
# Exit when: Choppiness Index crosses back to neutral zone (38.2-61.8)
# Choppiness Index identifies market regime (range vs trend), EMA34 filters direction, volume confirms strength.
# Works in bull (buy range dips) and bear (sell trend rallies) by adapting to market conditions.
name = "4h_Choppiness_EMA34_Volume_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Choppiness Index and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period)
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR14 = smoothed TR (using Wilder's smoothing)
    atr14 = np.zeros_like(tr)
    atr14[13] = np.mean(tr[1:14])  # first ATR14 value
    for i in range(14, len(tr)):
        atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # Highest high and lowest low over 14 periods
    hh14 = np.zeros_like(high_1d)
    ll14 = np.zeros_like(low_1d)
    for i in range(len(high_1d)):
        if i < 13:
            hh14[i] = np.nan
            ll14[i] = np.nan
        else:
            hh14[i] = np.max(high_1d[i-13:i+1])
            ll14[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index = 100 * log10(sum(ATR14 over 14) / (HH14 - LL14)) / log10(14)
    chop = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if hh14[i] != ll14[i] and not np.isnan(atr14[i]):
            sum_atr14 = np.sum(atr14[i-13:i+1])
            chop[i] = 100 * np.log10(sum_atr14 / (hh14[i] - ll14[i])) / np.log10(14)
    
    # Calculate EMA34 on daily data for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1D data to 4H timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_1d_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Chop > 61.8 (range), price above EMA34, volume spike
            if (chop_val > 61.8 and price > ema34 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Chop < 38.2 (trend), price below EMA34, volume spike
            elif (chop_val < 38.2 and price < ema34 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Chop crosses back below 61.8 (leaving range)
            if chop_val < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Chop crosses back above 38.2 (leaving trend)
            if chop_val > 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals