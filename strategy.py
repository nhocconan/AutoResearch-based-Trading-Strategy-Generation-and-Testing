#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from daily timeframe for structure
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation (>1.3x 20-period EMA) filters for institutional participation
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years)
# Works in bull markets (price > daily EMA50 + break above R1 + volume) and bear markets (price < daily EMA50 + break below S1 + volume)
# Uses discrete position sizing (0.25) to balance return potential with drawdown control

name = "4h_Camarilla_R1S1_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    pivot = (df_1d_high + df_1d_low + df_1d_close) / 3.0
    r1 = df_1d_close + (df_1d_high - df_1d_low) * 1.1 / 12.0
    s1 = df_1d_close - (df_1d_high - df_1d_low) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R1 with bullish daily trend and volume confirmation
            if bullish_bias and close[i] > r1_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with bearish daily trend and volume confirmation
            elif bearish_bias and close[i] < s1_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S1 OR daily trend turns bearish
            if close[i] < s1_aligned[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R1 OR daily trend turns bullish
            if close[i] > r1_aligned[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals