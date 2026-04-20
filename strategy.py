#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian breakout + volume confirmation
# - Choppiness Index > 61.8 = range (mean revert to Donchian mid)
# - Choppiness Index < 38.2 = trend (breakout)
# - Works in bull/bear by adapting to market regime
# - Low trade frequency: only trade when clear regime + breakout align

name = "4h_Chop_Donchian_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Choppiness Index (needs daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === Daily Choppiness Index (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # ATR(14)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros_like(atr)
    for i in range(atr_period-1, len(atr)):
        if i == atr_period-1:
            sum_atr[i] = np.sum(atr[i-atr_period+1:i+1])
        else:
            sum_atr[i] = sum_atr[i-1] - atr[i-atr_period] + atr[i]
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(high_1d)
    min_low = np.zeros_like(low_1d)
    for i in range(len(high_1d)):
        if i < atr_period-1:
            max_high[i] = np.max(high_1d[:i+1])
            min_low[i] = np.min(low_1d[:i+1])
        else:
            max_high[i] = np.max(high_1d[i-atr_period+1:i+1])
            min_low[i] = np.min(low_1d[i-atr_period+1:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1d, 50.0)  # default neutral
    for i in range(atr_period-1, len(close_1d)):
        if sum_atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(atr_period)
    
    # Align Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Donchian Channels (20-period) ===
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Upper and lower bands
    upper = np.full_like(high_4h, np.nan)
    lower = np.full_like(low_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:  # 20 periods needed
            upper[i] = np.max(high_4h[i-19:i+1])
            lower[i] = np.min(low_4h[i-19:i+1])
    
    # Middle line (for exit)
    middle = (upper + lower) / 2
    
    # === Volume Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        chop_val = chop_aligned[i]
        upper_val = upper[i]
        lower_val = lower[i]
        middle_val = middle[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(chop_val) or np.isnan(upper_val) or 
            np.isnan(lower_val) or np.isnan(middle_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Range regime (chop > 61.8): mean revert at Donchian bands
            if chop_val > 61.8:
                if close_val <= lower_val and vol_ratio_val > 1.5:  # long at support
                    signals[i] = 0.25
                    position = 1
                elif close_val >= upper_val and vol_ratio_val > 1.5:  # short at resistance
                    signals[i] = -0.25
                    position = -1
            # Trend regime (chop < 38.2): breakout
            elif chop_val < 38.2:
                if close_val > upper_val and vol_ratio_val > 1.5:  # breakout long
                    signals[i] = 0.25
                    position = 1
                elif close_val < lower_val and vol_ratio_val > 1.5:  # breakdown short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: 
            # - In range: return to middle
            # - In trend: close below lower band OR chop > 61.8 (range resumption)
            if chop_val > 61.8:
                if close_val >= middle_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # trend regime
                if close_val < lower_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit:
            # - In range: return to middle
            # - In trend: close above upper band OR chop > 61.8 (range resumption)
            if chop_val > 61.8:
                if close_val <= middle_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # trend regime
                if close_val > upper_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals