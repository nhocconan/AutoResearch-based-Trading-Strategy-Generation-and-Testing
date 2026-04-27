# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter and volume spike
# Camarilla levels provide institutional support/resistance; breakouts with volume
# and higher timeframe trend capture momentum while minimizing false breaks.
# Works in bull/bear by filtering breakout direction with 1d EMA trend.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.
# Focus on ETH and BTC as primary markets.

#!/usr/bin/env python3
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous day)
    camarilla_H1 = np.full(len(df_1d), np.nan)
    camarilla_L1 = np.full(len(df_1d), np.nan)
    camarilla_H2 = np.full(len(df_1d), np.nan)
    camarilla_L2 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's range
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        # Camarilla levels (R1/S1 = H1/L1, R2/S2 = H2/L2)
        camarilla_H1[i] = pc + 1.1 * rang / 6
        camarilla_L1[i] = pc - 1.1 * rang / 6
        camarilla_H2[i] = pc + 1.1 * rang / 4
        camarilla_L2[i] = pc - 1.1 * rang / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d close)
    H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    H2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 2.0 x 24-period average (12 days of 12h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (1 bar), EMA (50), volume MA (24)
    start_idx = max(1, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or 
            np.isnan(H2_aligned[i]) or np.isnan(L2_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: break above H1 with volume and bullish trend
            if price > H1_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below L1 with volume and bearish trend
            elif price < L1_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to L2 (mean reversion) or trend turns bearish
            if price <= L2_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to H2 (mean reversion) or trend turns bullish
            if price >= H2_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0