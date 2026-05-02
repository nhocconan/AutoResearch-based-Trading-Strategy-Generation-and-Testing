#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) derived from daily OHLC provide strong support/resistance
# Breakout above R3 or below S3 with volume confirmation indicates institutional participation
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.30) to minimize fee churn and control drawdown
# Works in bull markets (breakout above R3 + 1d trend up) and bear markets (breakout below S3 + 1d trend down)

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # 1d data for trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for each 1d bar (based on same day's OHLC)
    # Standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 4
    camarilla_s3 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (use same day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume confirmation and 1d trend up
            if close[i] > camarilla_r3_aligned[i] and bullish_bias and volume_confirmation[i]:
                signals[i] = 0.30
                position = 1
            # Short: Breakout below S3 with volume confirmation and 1d trend down
            elif close[i] < camarilla_s3_aligned[i] and bearish_bias and volume_confirmation[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 (reversal) OR 1d trend turns bearish
            if close[i] < camarilla_s3_aligned[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 (reversal) OR 1d trend turns bullish
            if close[i] > camarilla_r3_aligned[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals