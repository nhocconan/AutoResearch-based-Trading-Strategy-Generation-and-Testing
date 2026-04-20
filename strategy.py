# 1d_1w_Camarilla_R1S1_Breakout_Volume
# Hypothesis: Daily price breaking above Camarilla R1 or below S1 with weekly trend filter and volume confirmation
# Works in bull markets via breakout continuation and in bear via mean reversion at extreme levels
# Uses weekly trend to avoid counter-trend trades, volume to confirm breakout strength
# Target: 20-50 trades/year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === Daily: Calculate Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Weekly: Trend filter (EMA20) ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Daily: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_weekly = ema20_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema_weekly) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with weekly uptrend and volume confirmation
            if (close_val > r1_val and          # Price breaks above Camarilla R1
                close_val > ema_weekly and      # Weekly uptrend filter
                vol_ratio_val > 1.5):           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with weekly downtrend and volume confirmation
            elif (close_val < s1_val and        # Price breaks below Camarilla S1
                  close_val < ema_weekly and    # Weekly downtrend filter
                  vol_ratio_val > 1.5):         # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Camarilla center (P) or weekly trend breaks
            camarilla_p = (high_1d[i] + low_1d[i] + close_1d[i]) / 3  # Daily pivot point
            if (close_val < camarilla_p or      # Price returns to pivot
                close_val < ema_weekly):        # Weekly trend breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Camarilla center (P) or weekly trend breaks
            camarilla_p = (high_1d[i] + low_1d[i] + close_1d[i]) / 3  # Daily pivot point
            if (close_val > camarilla_p or      # Price returns to pivot
                close_val > ema_weekly):        # Weekly trend breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals