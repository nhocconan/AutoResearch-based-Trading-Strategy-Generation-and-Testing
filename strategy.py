#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance levels
# Breakout above R3 or below S3 with volume confirms institutional participation
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
# Designed for 4h timeframe targeting 19-50 trades/year (75-200 total over 4 years)
# Works in bull markets (breakout above R3 + daily uptrend) and bear markets (breakdown below S3 + daily downtrend)
# Uses discrete position sizing (0.30) to balance return potential with drawdown control

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla width = (high - low) * 1.1 / 12
    camarilla_width = (df_1d['high'] - df_1d['low']) * 1.1 / 12
    # R3 = close + (width * 1.1)
    r3 = df_1d['close'] + (camarilla_width * 1.1)
    # S3 = close - (width * 1.1)
    s3 = df_1d['close'] - (camarilla_width * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation (>1.5x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 with volume confirmation and daily uptrend
            if close[i] > r3_aligned[i] and volume_confirmation[i] and bullish_bias:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S3 with volume confirmation and daily downtrend
            elif close[i] < s3_aligned[i] and volume_confirmation[i] and bearish_bias:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 OR daily trend turns bearish
            if close[i] < s3_aligned[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 OR daily trend turns bullish
            if close[i] > r3_aligned[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals