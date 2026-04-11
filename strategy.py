#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index + 1d Donchian breakout + volume filter
# - Choppiness Index (CHOP) measures market regime: >61.8 = ranging (mean revert), <38.2 = trending
# - In ranging markets (CHOP > 61.8): fade at 1d Donchian bands (sell at upper band, buy at lower band)
# - In trending markets (CHOP < 38.2): breakout continuation (buy upper band break, sell lower band break)
# - Volume confirmation: require volume > 1.5x 20-period average to avoid false breakouts
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in both bull (trend continuation) and bear (mean reversion in ranges) markets

name = "6h_1d_chop_donchian_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Donchian bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Pre-compute 6h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR over n) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop_raw = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # Neutral when no range
    
    # Pre-compute 6h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(chop_raw[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Regime detection
        chop = chop_raw[i]
        ranging_market = chop > 61.8  # Choppy/ranging market
        trending_market = chop < 38.2  # Trending market
        
        # Donchian breakout levels
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if ranging_market and vol_confirm:
            # In ranging markets: mean reversion at Donchian bands
            if price_low <= lower_band:  # Touched or broke lower band -> long
                enter_long = True
            if price_high >= upper_band:  # Touched or broke upper band -> short
                enter_short = True
        elif trending_market and vol_confirm:
            # In trending markets: breakout continuation
            if price_close > upper_band:  # Close above upper band -> long
                enter_long = True
            if price_close < lower_band:  # Close below lower band -> short
                enter_short = True
        
        # Exit conditions: opposite signal or volatility expansion
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if: short signal OR price returns to opposite band in ranging market
            exit_long = enter_short or (ranging_market and price_high >= upper_band)
        elif position == -1:
            # Exit short if: long signal OR price returns to opposite band in ranging market
            exit_short = enter_long or (ranging_market and price_low <= lower_band)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals