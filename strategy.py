#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 1-day timeframe for institutional reference points
# Breakouts at R1/S1 with volume confirmation capture institutional participation in trend direction
# 1d EMA50 filter ensures alignment with daily trend to avoid counter-trend whipsaws
# Designed for low trade frequency: ~15-30 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via mean reversion at extremes
# BTC/ETH focused: avoids SOL-only bias by requiring volume spike and trend alignment

name = "4h_Camarilla_R1S1_1dEMA50_Trend_Volume_v1"
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
    
    # 1d HTF data for Camarilla pivots, EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels: R1, R2, S1, S2
    # R1 = close + range * 1.1 / 12
    # R2 = close + range * 1.1 / 6
    # S1 = close - range * 1.1 / 12
    # S2 = close - range * 1.1 / 6
    camarilla_r1 = close_1d + daily_range * 1.1 / 12
    camarilla_r2 = close_1d + daily_range * 1.1 / 6
    camarilla_s1 = close_1d - daily_range * 1.1 / 12
    camarilla_s2 = close_1d - daily_range * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA (moderate filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient 1d data for EMA/Camarilla calculation
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R1 with volume spike
                if close[i] > camarilla_r1_aligned[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S1 with volume spike
                if close[i] < camarilla_s1_aligned[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S1 or price below 1d EMA50
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R1 or price above 1d EMA50
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals