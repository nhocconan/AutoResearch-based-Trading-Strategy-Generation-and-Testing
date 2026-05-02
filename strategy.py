#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability reversal/breakout zones
# 1d EMA34 ensures alignment with higher-timeframe trend to avoid counter-trend trades
# Volume spike (>1.8 x 20-period EMA) confirms breakout validity and reduces false signals
# Discrete position sizing (0.25) minimizes fee churn and controls drawdown
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost
# Works in bull markets (breakout above R1 + uptrend) and bear markets (breakdown below S1 + downtrend)

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.8 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_20)
    
    # 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need previous day data)
    start_idx = 1
    
    for i in range(start_idx, n):
        # Need previous day's OHLC for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Check for NaN in critical values
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirmation[i]) or
            np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
        #          R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
        #          S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low)
        #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_hl = prev_high - prev_low
        
        # Avoid division by zero or invalid ranges
        if range_hl <= 0:
            signals[i] = 0.0
            continue
            
        r1 = prev_close + 0.275 * range_hl
        s1 = prev_close - 0.275 * range_hl
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R1 with volume confirmation and uptrend
            if close[i] > r1 and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation and downtrend
            elif close[i] < s1 and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below R1 (failed breakout) OR trend changes to downtrend
            if close[i] < r1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above S1 (failed breakdown) OR trend changes to uptrend
            if close[i] > s1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals