#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter combined with Donchian breakout
# Uses daily trend filter (EMA50) to avoid counter-trend trades
# Long: Price breaks above Donchian(20) high + Choppiness < 38.2 (trending) + close > daily EMA50
# Short: Price breaks below Donchian(20) low + Choppiness < 38.2 (trending) + close < daily EMA50
# Exit: Price returns to Donchian midpoint or trend reverses
# Designed for low trade frequency (15-25/year) with strong trend capture in both bull/bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) - 20-period high/low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate Choppiness Index on 4h data
    atr_period = 14
    chop_period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Sum of ATR over chop_period
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Max(high) - Min(low) over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    range_max_min = max_high - min_low
    
    # Choppiness Index
    cpi = np.where(
        (range_max_min != 0) & (sum_atr != 0),
        100 * np.log10(sum_atr / range_max_min) / np.log10(chop_period),
        50.0
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(cpi[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        chop = cpi[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high + trending market (low chop) + uptrend
            if (price > donch_high[i] and 
                chop < 38.2 and 
                price > ema_val):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + trending market (low chop) + downtrend
            elif (price < donch_low[i] and 
                  chop < 38.2 and 
                  price < ema_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: return to Donchian midpoint or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to midpoint or trend turns down
                if (price < donch_mid[i] or 
                    price < ema_val):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to midpoint or trend turns up
                if (price > donch_mid[i] or 
                    price > ema_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Choppiness_Donchian_Trend"
timeframe = "4h"
leverage = 1.0