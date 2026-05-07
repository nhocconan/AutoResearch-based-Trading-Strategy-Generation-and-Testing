#!/usr/bin/env python3

name = "4h_Trix_Signal_With_Volume_And_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d TRIX (15,9,9)
    close_1d = df_1d['close'].values
    # First EMA
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False).mean()
    # Second EMA
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    # Third EMA
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    # TRIX = % change of third EMA
    trix_raw = ((ema3 / ema3.shift(1)) - 1) * 100
    trix = trix_raw.values
    # Signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False).mean().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume filter: current volume > 1.3x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day for 4h to reduce trades
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TRIX crosses above signal line with volume in 1d uptrend
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TRIX crosses below signal line with volume in 1d downtrend
            elif (trix_aligned[i] < trix_signal_aligned[i] and 
                  trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TRIX crosses below signal line or trend change
            if (trix_aligned[i] < trix_signal_aligned[i] and 
                trix_aligned[i-1] >= trix_signal_aligned[i-1]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above signal line or trend change
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX momentum with signal line crossovers on 1d timeframe, combined with 1d EMA34 trend filter and volume confirmation on 4h timeframe.
# Long when TRIX crosses above its signal line in a 1d uptrend with volume confirmation.
# Short when TRIX crosses below its signal line in a 1d downtrend with volume confirmation.
# Exits when TRIX crosses back in the opposite direction or trend changes.
# Uses 1d timeframe for signal generation to avoid noise, 4h for execution timing.
# Volume filter prevents false signals. Cooldown reduces trade frequency.
# Target: 25-40 trades/year. Works in bull markets by capturing momentum in uptrends
# and in bear markets by shorting momentum in downtrends. TRIX is effective at
# identifying trend changes and momentum shifts, making it suitable for both
# bull and bear markets when combined with trend and volume filters.