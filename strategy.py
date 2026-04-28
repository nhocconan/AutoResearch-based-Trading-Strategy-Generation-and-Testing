#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot R4/S4 breakout with 1w EMA34 trend filter and volume confirmation.
# Enter long when price breaks above 1w Camarilla R4 with volume spike and above 1w EMA34.
# Enter short when price breaks below 1w Camarilla S4 with volume spike and below 1w EMA34.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 7-25 trades/year.
# 1w timeframe provides stable structure, volume confirms breakout strength, EMA34 filters intermediate trend.
# Works in bull (breakouts with trend) and bear (failed breaks reverse) markets.

name = "1d_Camarilla_R4S4_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivots and EMA34 (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivots (using previous bar's high, low, close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    camarilla_r4 = np.full(n_1w, np.nan)
    camarilla_s4 = np.full(n_1w, np.nan)
    
    for i in range(1, n_1w):
        # Use previous bar to avoid look-ahead
        phigh = high_1w[i-1]
        plow = low_1w[i-1]
        pclose = close_1w[i-1]
        pivot = (phigh + plow + pclose) / 3.0
        rng = phigh - plow
        camarilla_r4[i] = pivot + rng * 1.1 / 2.0  # R4 level
        camarilla_s4[i] = pivot - rng * 1.1 / 2.0  # S4 level
    
    # Forward fill Camarilla levels
    camarilla_r4 = pd.Series(camarilla_r4).ffill().values
    camarilla_s4 = pd.Series(camarilla_s4).ffill().values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w indicators to 1d timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w EMA34
        above_ema = close[i] > ema_34_1w_aligned[i]
        below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r4_aligned[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s4_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < camarilla_s4_aligned[i] or below_ema
        short_exit = close[i] > camarilla_r4_aligned[i] or above_ema
        
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