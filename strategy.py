#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla levels from 1d timeframe (more stable than 4h) for structural support/resistance
# Breakout of R3 (bullish) or S3 (bearish) with volume > 2.0x 20-period average confirms institutional participation
# 1d EMA34 filter ensures trades only in direction of higher timeframe trend, reducing whipsaws in ranging markets
# Discrete position sizing (0.25) limits drawdown during 2022-style crashes while maintaining profit potential
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Camarilla pivots provide mathematically derived levels that work across market regimes
# Volume filter ensures breakouts have conviction, avoiding false breaks
# 1d EMA34 trend filter adds robustness by aligning with dominant trend, effective in both bull and bear markets

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop - MANDATORY for MTF
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.0*(high-low), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for 1d EMA34 and 4h volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 + volume confirm + price > 1d EMA34 (uptrend)
            if close[i] > camarilla_r3_aligned[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S3 + volume confirm + price < 1d EMA34 (downtrend)
            elif close[i] < camarilla_s3_aligned[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 (reversal to downside) or price < 1d EMA34 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 (reversal to upside) or price > 1d EMA34 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals