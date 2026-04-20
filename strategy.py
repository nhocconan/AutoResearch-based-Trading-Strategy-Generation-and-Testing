#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Works in bull (breakouts) and bear (fades false breakouts via trend filter)
# Target: 20-40 trades/year to avoid fee drag
name = "4h_Donchian20_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h: Trend filter (EMA34) ===
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # === 4h: Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema_trend = ema34_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(donch_high_val) or np.isnan(donch_low_val) or np.isnan(ema_trend) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + uptrend + volume
            if (close_val > donch_high_val and      # Breakout above 20-period high
                close_val > ema_trend and           # Price above 12h EMA34 (uptrend)
                vol_ratio_val > 1.5):               # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + downtrend + volume
            elif (close_val < donch_low_val and     # Breakdown below 20-period low
                  close_val < ema_trend and         # Price below 12h EMA34 (downtrend)
                  vol_ratio_val > 1.5):             # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend change
            if (close_val < donch_low_val or      # Break below 20-period low
                close_val < ema_trend):           # Price below 12h EMA34 (trend change)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend change
            if (close_val > donch_high_val or     # Break above 20-period high
                close_val > ema_trend):           # Price above 12h EMA34 (trend change)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals