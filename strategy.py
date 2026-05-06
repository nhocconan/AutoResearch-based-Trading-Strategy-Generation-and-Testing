#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Long when close > R1 AND close > 1d EMA34 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when close < S1 AND close < 1d EMA34 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide intraday support/resistance; 1d EMA34 ensures higher-timeframe trend alignment
# Volume spike confirms institutional participation; pivot exit works in ranging markets

name = "12h_Camarilla_R1S1_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), 
    # R2 = close + 0.75*(high-low), R1 = close + 0.5*(high-low)
    # PP = (high+low+close)/3, S1 = close - 0.5*(high-low), 
    # S2 = close - 0.75*(high-low), S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r1 = close + 0.5 * (high - low)
    camarilla_s1 = close - 0.5 * (high - low)
    camarilla_pp = (high + low + close) / 3.0
    
    # Shift to get previous bar's levels (no look-ahead)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_pp_prev = np.roll(camarilla_pp, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan
    camarilla_pp_prev[0] = np.nan
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r1_prev[i]) or np.isnan(camarilla_s1_prev[i]) or 
            np.isnan(camarilla_pp_prev[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout with trend and volume filters
            # Long: close > R1 AND uptrend AND volume spike
            if close[i] > camarilla_r1_prev[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < S1 AND downtrend AND volume spike
            elif close[i] < camarilla_s1_prev[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point (mean reversion)
            if close[i] <= camarilla_pp_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point (mean reversion)
            if close[i] >= camarilla_pp_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals