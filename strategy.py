#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme reversal with 1w EMA50 trend filter and volume spike confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Trades only when %R reaches extreme levels (< -80 for oversold long, > -20 for overbought short)
# Confirmed by 1w EMA50 trend direction and volume spike for institutional participation
# Target: 7-25 trades/year (30-100 over 4 years) to minimize fee drag
# Works in bull/bear by trading reversals against the 1w trend during extreme conditions

name = "1d_WilliamsR_Extreme_Reversal_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20, 14)  # Need sufficient history for 1w EMA, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals: extreme readings
        oversold = williams_r[i] < -80  # Oversold condition for long
        overbought = williams_r[i] > -20  # Overbought condition for short
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Oversold, volume spike, and uptrend (buy the dip in uptrend)
            if oversold and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Overbought, volume spike, and downtrend (sell the rally in downtrend)
            elif overbought and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when no longer oversold or trend reverses
            if williams_r[i] > -50 or not uptrend:  # Exit on recovery or trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when no longer overbought or trend reverses
            if williams_r[i] < -50 or not downtrend:  # Exit on recovery or trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals