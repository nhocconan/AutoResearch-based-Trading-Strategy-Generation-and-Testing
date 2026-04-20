#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian_Breakout_Trend_HTF"
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
    
    # === 12h: Trend filter (EMA50) ===
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 4h: Donchian channel (20) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Upper band: highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        ema50_trend = ema50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_band) or np.isnan(lower_band) or 
            np.isnan(ema50_trend) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band + uptrend + volume
            if (close_val > upper_band and    # Donchian breakout up
                close_val > ema50_trend and   # Price above 12h EMA50 (uptrend)
                vol_ratio_val > 1.5):         # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band + downtrend + volume
            elif (close_val < lower_band and  # Donchian breakout down
                  close_val < ema50_trend and # Price below 12h EMA50 (downtrend)
                  vol_ratio_val > 1.5):       # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below lower band (mean reversion)
            if close_val < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above upper band (mean reversion)
            if close_val > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals