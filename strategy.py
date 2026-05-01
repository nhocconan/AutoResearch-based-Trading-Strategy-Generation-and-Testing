#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide intraday support/resistance. Breakout above R3 or below S3 with volume confirmation.
# 1d EMA34 filter ensures we only trade in the direction of the daily trend: long when price > EMA34, short when price < EMA34.
# Works in bull (breakouts with volume in uptrend) and bear (breakouts with volume in downtrend).
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Daily HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily HTF data for Camarilla pivot calculation (using prior day to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use only completed daily bars
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    
    # Camarilla pivot levels for prior day
    pp_1d = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    r1_1d = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 12
    s1_1d = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 12
    r2_1d = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 6
    s2_1d = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 6
    r3_1d = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 4
    s3_1d = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 4
    r4_1d = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 2
    s4_1d = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: current volume > 2.0 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Daily EMA34 trend filter
        above_ema34 = curr_close > ema_34_1d_aligned[i]
        below_ema34 = curr_close < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_above_r3 = curr_close > r3_1d_aligned[i]
        breakout_below_s3 = curr_close < s3_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3, volume spike, above daily EMA34
            if breakout_above_r3 and vol_spike and above_ema34:
                signals[i] = 0.25
                position = 1
            # Short: break below S3, volume spike, below daily EMA34
            elif breakout_below_s3 and vol_spike and below_ema34:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below S3 or daily EMA34 failure
            if curr_close < s3_1d_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above R3 or daily EMA34 failure
            if curr_close > r3_1d_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals