#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla pivot levels (H4/L4) act as intraday support/resistance; breaks indicate strong momentum.
# 1d EMA34 ensures alignment with daily trend, reducing counter-trend whipsaws.
# Volume spike (>1.8x 20-bar average) confirms breakout validity.
# Designed for low-frequency, high-conviction entries on 12h timeframe.
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

name = "12h_Camarilla_H4L4_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Camarilla pivot calculation (using prior 1d OHLC)
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H4, L4 from prior 1d bar
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    camarilla_h4_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # 12h volume spike: >1.8x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient lookback for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_h4_1d_aligned[i]) or
            np.isnan(camarilla_l4_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions: price breaks Camarilla H4/L4
        long_breakout = close[i] > camarilla_h4_1d_aligned[i]
        short_breakout = close[i] < camarilla_l4_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit: price reverts to prior 1d close (mean reversion to daily VWAP-like level)
        long_exit = close[i] < camarilla_h4_1d_aligned[i] * 0.995  # 0.5% buffer to avoid whipsaw
        short_exit = close[i] > camarilla_l4_1d_aligned[i] * 1.005
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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