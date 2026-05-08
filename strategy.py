#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter with 12h EMA trend and volume confirmation
# Choppiness Index (CHOP) identifies ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In trending regimes (CHOP < 38.2), we follow 12h EMA direction with volume confirmation.
# In ranging regimes (CHOP > 61.8), we fade moves at 12h Bollinger Bands (2 std).
# This adapts to both bull/bear markets by switching between trend-following and mean-reversion.
# Targets 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

name = "6h_ChopRegime_EMA_BB"
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
    
    # Get 12h data for Choppiness Index calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness Index
    tr = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[1:atr_period])  # Skip first NaN
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR)/ (max(high)-min(low))) / log10(period)
    chop_period = 14
    chop = np.full_like(close_12h, np.nan)
    for i in range(chop_period, len(close_12h)):
        if not np.isnan(atr[i-chop_period+1:i+1]).any():
            sum_atr = np.nansum(atr[i-chop_period+1:i+1])
            max_high = np.nanmax(high_12h[i-chop_period+1:i+1])
            min_low = np.nanmin(low_12h[i-chop_period+1:i+1])
            if max_high > min_low and sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_period)
    
    # Chop regimes: >61.8 = ranging, <38.2 = trending
    chop_ranging = chop > 61.8
    chop_trending = chop < 38.2
    
    # Align Chop regimes to 6h timeframe
    chop_ranging_6h = align_htf_to_ltf(prices, df_12h, chop_ranging)
    chop_trending_6h = align_htf_to_ltf(prices, df_12h, chop_trending)
    
    # Get 12h EMA for trend direction
    ema_period = 21
    ema_12h = pd.Series(close_12h).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 12h Bollinger Bands for mean reversion
    bb_period = 20
    bb_std = 2.0
    sma_12h = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_12h = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_12h + bb_std * std_12h
    lower_bb = sma_12h - bb_std * std_12h
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    
    # Volume confirmation: volume > 1.5x 24-period MA (~12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_ranging_6h[i]) or np.isnan(chop_trending_6h[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if chop_trending_6h[i]:  # Trending regime: follow EMA
            if position == 0:
                # Enter long: price above EMA and volume confirmation
                if close[i] > ema_12h_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price below EMA and volume confirmation
                elif close[i] < ema_12h_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price crosses below EMA
                if close[i] < ema_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above EMA
                if close[i] > ema_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        elif chop_ranging_6h[i]:  # Ranging regime: fade at Bollinger Bands
            if position == 0:
                # Enter long: price at lower BB with volume confirmation
                if close[i] <= lower_bb_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price at upper BB with volume confirmation
                elif close[i] >= upper_bb_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price crosses above midline (SMA)
                if close[i] > sma_12h[-1] if len(sma_12h) > 0 else False:  # Simplified exit at midline
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses below midline
                if close[i] < sma_12h[-1] if len(sma_12h) > 0 else False:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Neutral regime: stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals