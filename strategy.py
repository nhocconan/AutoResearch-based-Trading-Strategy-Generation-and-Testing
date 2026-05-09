#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Bollinger Bands (20, 2) and volume confirmation.
# Enters long when price closes below lower BB with weekly uptrend and volume spike,
# short when price closes above upper BB with weekly downtrend and volume spike.
# Exits on trend reversal or price crossing back inside BB.
# Uses weekly timeframe for BB and trend to avoid look-ahead, daily for execution.
# Designed to work in both bull and bear markets by fading extremes in the direction of the weekly trend.
# Target: 15-30 trades/year to minimize fee drag.

name = "1d_BollingerBands_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on weekly close
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Weekly trend filter: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike filter: current volume > 2.0 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need enough data for BB (20) and EMA50 (50)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close below lower BB + weekly uptrend + volume spike
            if close[i] < lower and close[i] > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close above upper BB + weekly downtrend + volume spike
            elif close[i] > upper and close[i] < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back inside BB or weekly trend turns down
            if close[i] > lower or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back inside BB or weekly trend turns up
            if close[i] < upper or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals