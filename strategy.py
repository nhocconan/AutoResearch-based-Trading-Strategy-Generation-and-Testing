#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
    # Long: price breaks above H3 + volume > 1.5x 20-period avg + chop < 61.8
    # Short: price breaks below L3 + volume > 1.5x 20-period avg + chop < 61.8
    # Exit: price returns to pivot point (PP)
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    # Camarilla pivots work well in ranging markets; breakouts with volume confirm work in trending markets
    # Chop filter avoids false breakouts in choppy regimes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 4h data for primary timeframe (Camarilla pivots)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Get 12h data for volume confirmation and chop regime (MTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Calculate Camarilla pivot levels on 4h data (using previous bar's OHLC)
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 2
    # L3 = PP - (H - L) * 1.1 / 2
    typical_price = (high_4h + low_4h + close_4h) / 3
    pp = np.roll(typical_price, 1)  # previous bar's PP
    high_low_range = np.roll(high_4h - low_4h, 1)  # previous bar's range
    
    camarilla_pp = pp
    camarilla_h3 = pp + (high_low_range * 1.1 / 2)
    camarilla_l3 = pp - (high_low_range * 1.1 / 2)
    
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
    
    # Align all indicators to 4h timeframe (primary)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: chop < 61.8 indicates trending market (good for breakouts)
        is_trending_regime = chop_aligned[i] < 61.8
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume_12h[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = close_4h[i] > camarilla_h3_aligned[i]
        breakout_down = close_4h[i] < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = is_trending_regime and breakout_up and volume_confirmed
        enter_short = is_trending_regime and breakout_down and volume_confirmed
        
        # Exit conditions: price returns to Camarilla pivot point
        exit_long = position == 1 and close_4h[i] <= camarilla_pp_aligned[i]
        exit_short = position == -1 and close_4h[i] >= camarilla_pp_aligned[i]
        
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

name = "4h_12h_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0