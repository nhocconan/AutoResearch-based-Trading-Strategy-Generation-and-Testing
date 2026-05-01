#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -80 or > -20) with
# volume spike indicate potential reversals. 1d EMA34 ensures we trade with the daily trend.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) calculation
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R calculation on 12h timeframe (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 34  # Need 34 for EMA + 20 for volume MA + 14 for Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Williams %R extreme conditions
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike[i]
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold, volume spike, uptrend (buy dip in uptrend)
            if oversold and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought, volume spike, downtrend (sell rally in downtrend)
            elif overbought and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought or trend reversal
            if williams_r[i] > -20 or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold or trend reversal
            if williams_r[i] < -80 or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals