#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter + 1-day Donchian(20) breakout with volume confirmation
# Uses Choppiness Index to detect trending (CHOP < 38.2) vs ranging (CHOP > 61.8) markets
# In trending markets: trade breakouts of daily Donchian channels
# In ranging markets: fade extreme price levels near Donchian boundaries
# Volume confirmation ensures institutional participation
# Designed to work in both bull and bear markets by adapting to market regime
# Target: 50-150 total trades over 4 years (12-37/year)

name = "exp_13496_12h_chop_regime_1d_donchian_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CHOPPINESS_PERIOD = 14
CHOPPINESS_TRENDING = 38.2   # Below this = trending
CHOPPINESS_RANGING = 61.8    # Above this = ranging
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(atr1, atr2), atr3)
    
    # True Range sum over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    # Choppiness Index formula
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Choppiness Index from daily data
    chop = calculate_choppiness(high_1d, low_1d, close_1d, CHOPPINESS_PERIOD)
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Daily Donchian channels
    highest_high_1d = pd.Series(high_1d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    highest_high_1d_align = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_align = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    
    # Daily volume MA
    volume_ma_1d = pd.Series(vol_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_align = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Daily ATR
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_align = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CHOPPINESS_PERIOD, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(chop_align[i]) or np.isnan(highest_high_1d_align[i]) or 
            np.isnan(lowest_low_1d_align[i]) or np.isnan(volume_ma_1d_align[i]) or
            np.isnan(atr_1d_align[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using daily volume MA aligned to 12h)
        volume_ok = volume[i] > (volume_ma_1d_align[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1d_align[i]) else False
        
        # Regime filter
        is_trending = chop_align[i] < CHOPPINESS_TRENDING
        is_ranging = chop_align[i] > CHOPPINESS_RANGING
        
        # Price position relative to daily Donchian channels
        price_position = (close[i] - lowest_low_1d_align[i]) / (highest_high_1d_align[i] - lowest_low_1d_align[i]) if (highest_high_1d_align[i] - lowest_low_1d_align[i]) > 0 else 0.5
        
        # Generate signals based on regime
        if position == 0:
            if is_trending and volume_ok:
                # Trending market: trade breakouts
                breakout_up = close[i] > highest_high_1d_align[i]
                breakout_down = close[i] < lowest_low_1d_align[i]
                
                if breakout_up:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_1d_align[i])
                elif breakout_down:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_1d_align[i])
                    
            elif is_ranging and volume_ok:
                # Ranging market: fade extremes
                fade_low = price_position < 0.1  # Near lower band
                fade_high = price_position > 0.9  # Near upper band
                
                if fade_low:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_1d_align[i])
                elif fade_high:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_1d_align[i])
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals