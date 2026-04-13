#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h timeframe with 12h HTF filter
    # Long: price breaks above 12h Donchian(20) high + volume > 1.3x 20-period avg + chop < 61.8 (trending)
    # Short: price breaks below 12h Donchian(20) low + volume > 1.3x 20-period avg + chop < 61.8 (trending)
    # Exit: price returns to 12h Donchian middle
    # Using 4h timeframe targets 30-60 trades/year to minimize fee drag
    # Donchian breakouts with volume/regime confirmation work in both bull/bear markets
    # Added: ATR-based volatility filter to avoid choppy markets and reduce false breakouts
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for HTF Donchian channels and regime filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Calculate Donchian channels on 12h data (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate Chopiness Index on 12h data (14-period)
    def calculate_chop(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        
        # Sum of True Range over window
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Chop = log10(atr_sum / (highest_high - lowest_low)) / log10(window) * 100
        highest_low_diff = highest_high - lowest_low
        chop = np.where(
            (highest_low_diff > 0) & (~np.isnan(atr_sum)),
            np.log10(atr_sum / highest_low_diff) / np.log10(window) * 100,
            50  # default to middle when invalid
        )
        return chop
    
    chop = calculate_chop(high_12h, low_12h, close_12h, window=14)
    
    # Volume averages on 12h data (20-period)
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR on 12h data (14-period) for volatility filter
    def calculate_atr(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr
    
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, window=14)
    atr_ma_20 = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(atr_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: chop < 61.8 indicates trending market (good for breakouts)
        is_trending_regime = chop_aligned[i] < 61.8
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        volume_confirmed = volume_12h[i] > 1.3 * vol_avg_20_12h_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods (choppy markets)
        vol_filter = atr_12h[i] > 0.5 * atr_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_up = close_12h[i] > donchian_high_aligned[i]
        breakout_down = close_12h[i] < donchian_low_aligned[i]
        
        # Entry conditions
        enter_long = is_trending_regime and breakout_up and volume_confirmed and vol_filter
        enter_short = is_trending_regime and breakout_down and volume_confirmed and vol_filter
        
        # Exit conditions: price returns to 12h Donchian middle
        exit_long = position == 1 and close_12h[i] <= donchian_mid_aligned[i]
        exit_short = position == -1 and close_12h[i] >= donchian_mid_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_volume_chop_atr_v1"
timeframe = "4h"
leverage = 1.0