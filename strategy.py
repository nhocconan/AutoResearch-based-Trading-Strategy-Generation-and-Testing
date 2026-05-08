#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1w ADX trend filter + volume spike confirmation.
# Long when 12h Choppiness > 61.8 (range) AND price reverses from lower Bollinger Band (20,2) AND volume > 2x 20-period average.
# Short when 12h Choppiness > 61.8 (range) AND price reverses from upper Bollinger Band (20,2) AND volume > 2x 20-period average.
# Exit when price crosses the 20-period SMA.
# This strategy captures mean reversion in ranging markets with volatility filters, suitable for both bull and bear regimes.
# Choppiness Index identifies ranging markets (avoids trend-following whipsaw), Bollinger Bands provide reversal signals,
# volume confirmation reduces false signals, and 1w ADX ensures we avoid strong trends where mean reversion fails.

name = "12h_Choppiness_Bollinger_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Choppiness Index and Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Choppiness Index (14-period)
    def choppiness_index(high, low, close, window=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        # Sum of true range over window
        sum_tr = np.nansum(pd.Series(atr).rolling(window=window, min_periods=window).values, axis=1)
        # Highest high and lowest low over window
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        chop = np.zeros_like(close)
        mask = (range_hl != 0) & ~np.isnan(sum_tr)
        chop[mask] = 100 * np.log10(sum_tr[mask] / range_hl[mask]) / np.log10(window)
        chop[~mask] = 50.0  # default to middle when undefined
        return chop
    
    chop = choppiness_index(high_12h, low_12h, close_12h, 14)
    chop_align = align_htf_to_ltf(prices, df_12h, chop)
    
    # Bollinger Bands (20,2) on 12h close
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_align = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_align = align_htf_to_ltf(prices, df_12h, lower_bb)
    sma_20_align = align_htf_to_ltf(prices, df_12h, sma_20)
    
    # 1w ADX for trend filter (avoid strong trends)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ADX calculation (14-period)
    def adx(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # first period has no previous close
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        dm_plus_smooth = pd.Series(dm_plus).rolling(window=window, min_periods=window).mean().values
        dm_minus_smooth = pd.Series(dm_minus).rolling(window=window, min_periods=window).mean().values
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
        minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_vals = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
        return adx_vals
    
    adx_1w = adx(high_1w, low_1w, close_1w, 14)
    adx_1w_align = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(chop_align[i]) or np.isnan(upper_bb_align[i]) or np.isnan(lower_bb_align[i]) or
            np.isnan(sma_20_align[i]) or np.isnan(adx_1w_align[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long when ranging (chop > 61.8), price at lower BB, volume spike, and weak trend (ADX < 25)
            long_cond = (chop_align[i] > 61.8) and (close[i] <= lower_bb_align[i]) and volume_filter[i] and (adx_1w_align[i] < 25)
            # Short when ranging (chop > 61.8), price at upper BB, volume spike, and weak trend (ADX < 25)
            short_cond = (chop_align[i] > 61.8) and (close[i] >= upper_bb_align[i]) and volume_filter[i] and (adx_1w_align[i] < 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above 20-period SMA
            if close[i] > sma_20_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below 20-period SMA
            if close[i] < sma_20_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals