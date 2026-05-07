#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with weekly trend filter and daily volume confirmation.
# Long when: Williams %R < -80 (oversold) AND weekly close > weekly EMA(34) (uptrend) AND daily volume > daily SMA(20) volume (confirmation)
# Short when: Williams %R > -20 (overbought) AND weekly close < weekly EMA(34) (downtrend) AND daily volume > daily SMA(20) volume (confirmation)
# Exit when Williams %R crosses back to -50.
# Designed for 6h timeframe with low trade frequency (target: 15-30/year) to avoid fee drag.
# Uses weekly for trend direction and daily for volume confirmation to avoid false signals.
# Works in bull markets via buying oversold dips in uptrend, in bear markets via selling overbought rallies in downtrend.
# Volume filter ensures participation and avoids low-liquidity whipsaws.
name = "6h_WilliamsR_WeeklyTrend_DailyVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Weekly EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_uptrend = close_1w > ema_34_1w
    weekly_downtrend = close_1w < ema_34_1w
    
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Daily volume confirmation: volume > SMA(20) volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > vol_sma_20
    
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 AND weekly uptrend AND volume confirmation
            long_condition = (williams_r[i] < -80) and weekly_uptrend_aligned[i] and volume_confirm_aligned[i]
            # Short: Williams %R > -20 AND weekly downtrend AND volume confirmation
            short_condition = (williams_r[i] > -20) and weekly_downtrend_aligned[i] and volume_confirm_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R > -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R < -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals