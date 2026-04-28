#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla pivot levels (R1/S1) with volume confirmation and weekly EMA34 trend filter.
# Enter long when price breaks above weekly R1 with volume spike and above weekly EMA34.
# Enter short when price breaks below weekly S1 with volume spike and below weekly EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# This pattern has proven ETH/SOL edge on lower timeframes and adapts to 1d/1w for BTC/ETH edge in both bull and bear markets.

name = "1d_Camarilla_R1S1_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Get weekly data for Camarilla pivots and EMA
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    R1 = np.full(n_1w, np.nan)
    S1 = np.full(n_1w, np.nan)
    PP = np.full(n_1w, np.nan)
    
    for i in range(n_1w):
        # Camarilla pivot calculation
        PP[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        range_1w = high_1w[i] - low_1w[i]
        R1[i] = PP[i] + range_1w * 1.1 / 12.0
        S1[i] = PP[i] - range_1w * 1.1 / 12.0
    
    # Forward fill to get most recent pivot levels
    R1 = pd.Series(R1).ffill().values
    S1 = pd.Series(S1).ffill().values
    PP = pd.Series(PP).ffill().values
    
    # Align 1w Camarilla levels to 1d timeframe with 1-bar delay for confirmation
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA34
        above_ema = close[i] > ema_34_1w_aligned[i]
        below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R1_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S1_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite pivot level or trend reversal
        long_exit = close[i] < S1_aligned[i] or below_ema
        short_exit = close[i] > R1_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals