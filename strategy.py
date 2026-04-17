#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot S1/R1 breakout with 1w trend filter and volume confirmation
# Works in bull markets (breakouts continue) and bear markets (mean reversion at extremes)
# Target: 15-25 trades/year, low frequency to avoid fee drag
# Edge: Institutional pivot levels + trend alignment + volume confirmation reduces false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to daily
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Camarilla pivot points (using previous day's OHLC)
    # Camarilla: R1 = close + 0.115*(high-low), S1 = close - 0.115*(high-low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r1 = prev_close + 0.115 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.115 * (prev_high - prev_low)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 35  # Need EMA34 warmup and Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with weekly uptrend and volume
            if (close[i] > camarilla_r1[i] and 
                ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and  # Rising trend
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with weekly downtrend and volume
            elif (close[i] < camarilla_s1[i] and 
                  ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and  # Falling trend
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Camarilla S1 (mean reversion)
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Camarilla R1 (mean reversion)
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_S1R1_Breakout_EMA34Trend_Volume"
timeframe = "1d"
leverage = 1.0