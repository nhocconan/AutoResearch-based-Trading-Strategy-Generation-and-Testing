#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Pivot_R1S1_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly: Trend filter (EMA 20) ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily: Pivot points (previous day) ===
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily: ATR for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        close_val = close_1d[i]
        r1_level = r1[i]
        s1_level = s1[i]
        ema_20_val = ema_20_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_level) or np.isnan(s1_level) or np.isnan(ema_20_val) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = np.nanmedian(atr[max(0, i-49):i+1]) if i >= 1 else np.nan
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation, volatility filter, and weekly uptrend
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 2.0 and    # Strong volume confirmation
                vol_filter and             # Volatility filter
                close_val > ema_20_val):   # Weekly uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation, volatility filter, and weekly downtrend
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 2.0 and    # Strong volume confirmation
                  vol_filter and             # Volatility filter
                  close_val < ema_20_val):   # Weekly downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 (reversion to mean)
            if close_val < r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 (reversion to mean)
            if close_val > s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals