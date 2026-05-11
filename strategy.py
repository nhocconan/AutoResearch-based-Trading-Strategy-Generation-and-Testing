#%%
#!/usr/bin/env python3
# 1D_Weekly_Trend_Follow_v1: Trend-following strategy using weekly higher timeframe trend (1w) and daily breakouts with volume confirmation.
# Hypothesis: In trending markets (both bull and bear), price tends to continue in the direction of the weekly trend. 
# Breakouts above/below prior day's high/low with volume confirmation capture momentum. Works in bull (up-trends) and bear (down-trends) by following weekly trend.
# Target: 20-50 trades per year to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn.

name = "1D_Weekly_Trend_Follow_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for weekly trend
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (primary HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend (responsive but smooth)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily high/low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's high/low (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day's high
    prev_low[0] = low_1d[0]    # First day uses same day's low
    
    # Align daily levels to 1d timeframe (no shift needed as same timeframe)
    prev_high_aligned = prev_high  # Already aligned to daily
    prev_low_aligned = prev_low    # Already aligned to daily
    
    # Volume filter: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above prior day's high with volume AND weekly trend is up
            if (close[i] > prev_high_aligned[i] and 
                volume_surge and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior day's low with volume AND weekly trend is down
            elif (close[i] < prev_low_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to prior day's opposite level or weekly trend fails
            if position == 1:
                # Exit long: price returns to prior day's low or weekly trend turns down
                if (close[i] < prev_low_aligned[i]) or (close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to prior day's high or weekly trend turns up
                if (close[i] > prev_high_aligned[i]) or (close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

#%%