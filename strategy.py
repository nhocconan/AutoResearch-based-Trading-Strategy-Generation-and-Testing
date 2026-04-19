# 6h_WeeklyPivot_Breakout_Volume_ATR
# Hypothesis: 6h breakouts from weekly pivot levels (R1/S1) with volume confirmation and ATR volatility filter
# Weekly pivots provide strong support/resistance that holds across market regimes
# Volume confirms institutional participation, ATR filter avoids low-volatility false breakouts
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear via volatility-adjusted breakouts and volume confirmation

name = "6h_WeeklyPivot_Breakout_Volume_ATR"
timeframe = "6h"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    ph = df_1w['high'].shift(1).values  # Previous week high
    pl = df_1w['low'].shift(1).values   # Previous week low
    pc = df_1w['close'].shift(1).values # Previous week close
    
    # Weekly pivot calculations (standard floor trader pivots)
    pivot = (ph + pl + pc) / 3.0
    r1 = 2 * pivot - pl
    s1 = 2 * pivot - ph
    r2 = pivot + (ph - pl)
    s2 = pivot - (ph - pl)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # ATR(14) for volatility filter - calculated on 6h data
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        atr = np.full_like(close, np.nan)
        if len(close) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(close)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: only trade when ATR > 50% of its 50-period average (avoid low volatility)
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > (atr_ma[i] * 0.5) if not np.isnan(atr_ma[i]) else False
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or volatility drops
            if (close[i] < s1_aligned[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or volatility drops
            if (close[i] > r1_aligned[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals