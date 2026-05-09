#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter with 12h Donchian breakout.
# Choppiness Index > 61.8 indicates ranging market (mean reversion), < 38.2 indicates trending.
# In trending regime, take Donchian breakout in direction of trend; in ranging regime,
# take mean reversion at Donchian channel extremes. Uses 12h for trend/structure to avoid
# whipsaws. Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_Chop_DonchianBreakout_12hTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h high/low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower (20-period high/low)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Choppiness Index on 6h data (14-period)
    # CHOP = 100 * log10(sum(ATR over n) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    chop = np.where(hl_range > 0, 100 * np.log10(atr_sum / hl_range) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and other indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema_12h = ema_50_12h_aligned[i]
        chop_val = chop[i]
        curr_close = close[i]
        
        # Calculate volume spike for confirmation (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        vol_spike = volume[i] > 1.5 * vol_ma  # Require 1.5x average volume
        
        if position == 0:
            # Enter long: Donchian breakout above upper band AND uptrend (price > EMA50) AND volume spike
            # Only in trending regime (CHOP < 38.2) or strong breakout in any regime
            if (curr_close > donch_high_val and curr_close > ema_12h and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakdown below lower band AND downtrend (price < EMA50) AND volume spike
            elif (curr_close < donch_low_val and curr_close < ema_12h and vol_spike):
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging market: buy at support, sell at resistance
            elif chop_val > 61.8:  # Ranging market
                if curr_close <= donch_low_val and vol_spike:  # Near support, go long
                    signals[i] = 0.25
                    position = 1
                elif curr_close >= donch_high_val and vol_spike:  # Near resistance, go short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price returns to EMA50 OR Donchian lower band touched OR trend reversal
            if (curr_close <= ema_12h or curr_close <= donch_low_val or 
                chop_val > 61.8):  # Exit if market becomes ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to EMA50 OR Donchian upper band touched OR trend reversal
            if (curr_close >= ema_12h or curr_close >= donch_high_val or 
                chop_val > 61.8):  # Exit if market becomes ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals