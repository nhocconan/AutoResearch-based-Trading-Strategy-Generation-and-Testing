# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 12h timeframe with 1D Donchian breakout and volume confirmation reduces false signals and improves trend capture. 
# In bull markets, breakouts capture momentum; in bear markets, breakdowns avoid false reversals. Volume filter ensures 
# only significant moves are traded, reducing whipsaws and overtrading.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Calculate Donchian channels (20-period high/low) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Use rolling window for Donchian bands
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 12h: ATR for volatility filter ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_band) or np.isnan(lower_band) or np.isnan(vol_ratio_val) or 
            np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr[max(0, i-49):i+1]) if i >= 1 else np.nan
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        if position == 0:
            # Long: Price breaks above upper Donchian band with volume confirmation and volatility filter
            if (close_val > upper_band and   # Break above upper band
                vol_ratio_val > 2.0 and    # Strong volume confirmation
                vol_filter):               # Volatility filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band with volume confirmation and volatility filter
            elif (close_val < lower_band and   # Break below lower band
                  vol_ratio_val > 2.0 and    # Strong volume confirmation
                  vol_filter):               # Volatility filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below lower Donchian band (mean reversion)
            if close_val < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above upper Donchian band (mean reversion)
            if close_val > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals