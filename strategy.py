#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action filtered by 1d volatility regime and volume
# Uses 1d ATR ratio (ATR14 / ATR50) to identify high volatility regimes.
# Enters long when price closes above 6h high of previous 10 bars in high vol regime.
# Enters short when price closes below 6h low of previous 10 bars in high vol regime.
# Uses volume confirmation: current volume > 1.5x 20-period average.
# Designed to capture breakouts during volatile periods, effective in both bull and bear markets.
# Target: 15-30 trades/year.

name = "6h_VolatilityRegime_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr_daily = np.zeros(len(close_daily))
    atr14_daily = np.full(len(close_daily), np.nan)
    atr50_daily = np.full(len(close_daily), np.nan)
    
    for i in range(len(close_daily)):
        if i == 0:
            tr_daily[i] = high_daily[i] - low_daily[i]
        else:
            tr_daily[i] = max(
                high_daily[i] - low_daily[i],
                abs(high_daily[i] - close_daily[i-1]),
                abs(low_daily[i] - close_daily[i-1])
            )
        
        if i >= 13:  # ATR14
            if i == 13:
                atr14_daily[i] = np.mean(tr_daily[:14])
            else:
                atr14_daily[i] = (atr14_daily[i-1] * 13 + tr_daily[i]) / 14
        
        if i >= 49:  # ATR50
            if i == 49:
                atr50_daily[i] = np.mean(tr_daily[:50])
            else:
                atr50_daily[i] = (atr50_daily[i-1] * 49 + tr_daily[i]) / 50
    
    # Calculate ATR ratio (ATR14/ATR50) - high when > 0.8 indicates volatile regime
    atr_ratio_daily = np.full(len(close_daily), np.nan)
    for i in range(len(close_daily)):
        if not np.isnan(atr14_daily[i]) and not np.isnan(atr50_daily[i]) and atr50_daily[i] > 0:
            atr_ratio_daily[i] = atr14_daily[i] / atr50_daily[i]
    
    # Align daily data to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_daily, atr_ratio_daily)
    
    # Pre-calculate 6h rolling high/low of previous 10 bars
    high_10 = np.full(n, np.nan)
    low_10 = np.full(n, np.nan)
    if n >= 10:
        for i in range(10, n):
            high_10[i] = np.max(high[i-10:i])
            low_10[i] = np.min(low[i-10:i])
    
    # Pre-calculate volume average (20-period)
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 10)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(high_10[i]) or 
            np.isnan(low_10[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR ratio > 0.7 indicates elevated volatility
        vol_filter = atr_ratio_aligned[i] > 0.7
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: breakout of 10-bar range in high vol regime with volume
            if vol_filter and vol_confirm:
                if close[i] > high_10[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_10[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: close below 10-bar low or volatility drops
            if close[i] < low_10[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above 10-bar high or volatility drops
            if close[i] > high_10[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals