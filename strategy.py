#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Camarilla pivot levels provide strong intraday support/resistance (R3/S3 = extreme levels)
# Breakout above R3 or below S3 with volume confirmation indicates institutional participation
# 12h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades
# Designed for 4h timeframe targeting 20-50 trades/year (75-200 total over 4 years)
# Works in bull markets (breakout above R3 + 12h trend up) and bear markets (breakout below S3 + 12h trend down)
# Uses discrete position sizing (0.30) to balance return potential with drawdown control

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's OHLC)
    # Camarilla R3 = close + 1.1*(high - low)*1.1/4
    # Camarilla S3 = close - 1.1*(high - low)*1.1/4
    # Actually: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # But standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
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
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume confirmation and 12h trend up
            if close[i] > camarilla_r3_aligned[i] and bullish_bias and volume_confirmation[i]:
                signals[i] = 0.30
                position = 1
            # Short: Breakout below S3 with volume confirmation and 12h trend down
            elif close[i] < camarilla_s3_aligned[i] and bearish_bias and volume_confirmation[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 (reversal) OR 12h trend turns bearish
            if close[i] < camarilla_s3_aligned[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 (reversal) OR 12h trend turns bullish
            if close[i] > camarilla_r3_aligned[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals