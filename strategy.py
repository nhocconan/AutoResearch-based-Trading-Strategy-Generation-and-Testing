#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla R4/S4 levels (close ± (high-low)*1.1/2) represent stronger support/resistance than R3/S3
# Breakout above R4 or below S4 with volume confirmation indicates strong institutional participation
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades in ranging markets
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (breakout above R4 + 1w trend up) and bear markets (breakout below S4 + 1w trend down)

name = "12h_Camarilla_R4S4_Breakout_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for Camarilla pivot levels (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R4/S4 levels for each 1d bar (based on same day's OHLC)
    # Camarilla R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    camarilla_r4 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    camarilla_s4 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use same day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation - 20-period volume EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)  # Require 2x average volume for breakout
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R4 with volume confirmation and 1w trend up
            if close[i] > camarilla_r4_aligned[i] and bullish_bias and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S4 with volume confirmation and 1w trend down
            elif close[i] < camarilla_s4_aligned[i] and bearish_bias and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S4 (reversal) OR 1w trend turns bearish
            if close[i] < camarilla_s4_aligned[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R4 (reversal) OR 1w trend turns bullish
            if close[i] > camarilla_r4_aligned[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals