# 1d_1w_1d_CAMARILLA_MR_WITH_VOLUME_CONFIRMATION
# Hypothesis: Price reversals at Camarilla H4/L4 levels with volume confirmation provide mean-reversion opportunities.
# In ranging markets, price tends to revert from H4/L4 back toward the daily close. In trending markets,
# H4/L4 act as strong support/resistance where reversals still occur during pullbacks. Volume confirmation
# filters out weak reversals. Works in both bull/bear because mean reversion occurs at all market phases.
# Weekly EMA filter ensures we only take reversals in the direction of the higher timeframe trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(df_1w)):
            ema_50_1w[i] = (close_1w[i] - ema_50_1w[i-1]) * multiplier + ema_50_1w[i-1]
    
    # Align weekly EMA to 1d timeframe
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 1d timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d_aligned[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels based on previous day's range
        # Need previous day's data - use index-1 for daily data alignment
        if i >= 1:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            prev_range = prev_high - prev_low
            
            # Camarilla H4 and L4 levels
            h4 = prev_close + (prev_range * 1.1/2)
            l4 = prev_close - (prev_range * 1.1/2)
            
            # Align H4/L4 to 1d timeframe (constant values for the day)
            h4_1d = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), h4))[i]
            l4_1d = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), l4))[i]
            
            if position == 0:
                # Long: Price touches or goes below L4 and closes back above L4
                # with volume confirmation AND above weekly EMA50 (trend alignment)
                if low[i] <= l4 and close[i] > l4 and volume[i] > volume_ma[i] and close[i] > ema_50_1d[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price touches or goes above H4 and closes back below H4
                # with volume confirmation AND below weekly EMA50 (trend alignment)
                elif high[i] >= h4 and close[i] < h4 and volume[i] > volume_ma[i] and close[i] < ema_50_1d[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price reaches the previous day's close (mean reversion target)
                # or touches/goes above H4 (failure of mean reversion)
                if close[i] >= prev_close or high[i] >= h4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price reaches the previous day's close (mean reversion target)
                # or touches/goes below L4 (failure of mean reversion)
                if close[i] <= prev_close or low[i] <= l4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_1d_CAMARILLA_MR_WITH_VOLUME_CONFIRMATION"
timeframe = "1d"
leverage = 1.0