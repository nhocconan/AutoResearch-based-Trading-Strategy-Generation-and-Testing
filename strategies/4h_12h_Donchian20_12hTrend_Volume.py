#!/usr/bin/env python3
# 4h_12h_Donchian20_12hTrend_Volume
# Hypothesis: Uses 12h Donchian(20) breakout with 12h trend filter and volume confirmation on 4h timeframe.
# In bull markets: 12h uptrend + breakout above 20-period high captures momentum.
# In bear markets: 12h downtrend + breakdown below 20-period low captures short opportunities.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Target: 20-50 trades/year to minimize fee drag while capturing meaningful moves.

name = "4h_12h_Donchian20_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12-hour data for trend filter and Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h Donchian channels (20-period) from previous candle ---
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period high and low
    high_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 12h candle (avoid look-ahead)
    donchian_high = np.roll(high_max, 1)
    donchian_low = np.roll(low_min, 1)
    donchian_high[0] = np.nan  # First value invalid after roll
    donchian_low[0] = np.nan
    
    # Align 12h Donchian levels to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # --- 12h EMA50 for trend filter ---
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Volume confirmation (1.5x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 12h EMA50 (50 periods) and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume surge and 12h uptrend
            if (close[i] > donchian_high_aligned[i] and 
                volume_surge and 
                ema_50_12h_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume surge and 12h downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_surge and 
                  ema_50_12h_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below Donchian low OR 12h EMA50 turns down
                if (close[i] < donchian_low_aligned[i] or 
                    close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above Donchian high OR 12h EMA50 turns up
                if (close[i] > donchian_high_aligned[i] or 
                    close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals