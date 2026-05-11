# [Experiment #152770] 1d Donchian Breakout + 1w Trend + Volume Confirmation
# Hypothesis: Daily Donchian(20) breakouts in direction of weekly EMA50 trend with volume spike capture trends in both bull and bear markets.
# Uses 1d timeframe with 1h trend filter for regime context. Targets 10-25 trades/year via strict breakout + volume + trend conditions.
# Weekly trend filter avoids counter-trend trades in choppy markets. Volume confirmation ensures institutional participation.
# Exit on opposite Donchian break or trend reversal to capture full moves while limiting whipsaw.

name = "1d_Donchian_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA50 Trend Filter (HTF: 1w) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily Donchian Channels (20-period) ---
    # Calculate highest high and lowest low of past 20 days (excluding current)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # --- Volume Spike Detection (20-day average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 days for Donchian + 50 for EMA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ratio[i])):
            # Maintain current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above 20-day high + above weekly EMA50 + volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1d[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + below weekly EMA50 + volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1d[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price breaks below 20-day low OR trend turns down
                if (close[i] < low_20[i]) or (close[i] < ema_50_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above 20-day high OR trend turns up
                if (close[i] > high_20[i]) or (close[i] > ema_50_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals