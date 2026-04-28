#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Williams %R extremes with 1d trend filter and volume confirmation.
# Enter long when weekly %R < -80 (oversold) and price > 1d EMA34 with volume > 1.5x 20-bar average.
# Enter short when weekly %R > -20 (overbought) and price < 1d EMA34 with volume > 1.5x 20-bar average.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Weekly %R provides mean reversion edge in ranging markets, 1d EMA34 filters for trend alignment,
# volume confirmation ensures breakout/continuation strength. Works in bull (buy dips in uptrend) 
# and bear (sell rallies in downtrend) markets by trading extremes with trend filter.

name = "6h_WilliamsR_Weekly_Overextended_1dEMA34_Volume_v1"
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
    
    # Get weekly data for Williams %R (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate weekly Williams %R (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    williams_r = np.full(n_1w, np.nan)
    
    for i in range(14, n_1w):
        highest_high = np.max(high_1w[i-14:i+1])
        lowest_low = np.min(low_1w[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1w[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50.0
    
    # Align weekly Williams %R to 6h timeframe (with extra delay for indicator confirmation)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r, additional_delay_bars=1)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly Williams %R conditions with 1d trend filter and volume confirmation
        long_signal = williams_r_aligned[i] < -80 and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]
        short_signal = williams_r_aligned[i] > -20 and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]
        
        # Exit conditions: opposite Williams %R extreme or trend reversal
        long_exit = williams_r_aligned[i] > -20 or close[i] < ema_34_1d_aligned[i]
        short_exit = williams_r_aligned[i] < -80 or close[i] > ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
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