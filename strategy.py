#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_PriceChannel_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 12h Data (HTF) - Load ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Donchian Channel (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 20-period high/low for Donchian
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # === 4h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Simple trend filter: 50-period EMA
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(80, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        ema50_val = ema50[i]
        upper = donch_high_4h[i]
        lower = donch_low_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(ema50_val) or 
            np.isnan(upper) or np.isnan(lower)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume and trend filter
            if (close_val > upper and 
                vol_ratio_val > 1.8 and
                close_val > ema50_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume and trend filter
            elif (close_val < lower and 
                  vol_ratio_val > 1.8 and
                  close_val < ema50_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower Donchian or volume dries up
            if close_val < lower or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper Donchian or volume dries up
            if close_val > upper or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals