#!/usr/bin/env python3

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Classic formula: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # R3 = H + 1.1*(H-L), S3 = L - 1.1*(H-L)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_H4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_L4 = prev_close - 1.5 * (prev_high - prev_low)
    camarilla_R3 = prev_high + 1.1 * (prev_high - prev_low)
    camarilla_S3 = prev_low - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.8x 30-period average (4h)
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    vol_filter = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~2 days for 4h
    
    start_idx = max(35, 40)  # Warmup for daily data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R3 with volume in daily uptrend
            if (close[i] > R3_aligned[i] and 
                close[i-1] <= R3_aligned[i-1] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S3 with volume in daily downtrend
            elif (close[i] < S3_aligned[i] and 
                  close[i-1] >= S3_aligned[i-1] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below H4 OR trend change
            if (close[i] < H4_aligned[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above L4 OR trend change
            if (close[i] > L4_aligned[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with daily trend filter and volume confirmation.
# Long when price breaks above R3 level with volume spike in daily uptrend.
# Short when price breaks below S3 level with volume spike in daily downtrend.
# Exits on reversal to H4/L4 levels or trend change.
# Uses daily Camarilla levels for intraday support/resistance and EMA34 for trend.
# Volume confirmation filters false breakouts. Cooldown prevents overtrading.
# Target: 20-40 trades/year to avoid fee drift. Works in bull/bear by capturing
# significant intraday moves with trend alignment. Daily timeframe reduces noise.