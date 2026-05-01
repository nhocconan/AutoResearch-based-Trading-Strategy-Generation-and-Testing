#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power rising (less negative) with 1d EMA34 uptrend and volume spike
# Short when Bear Power < 0 and Bull Power falling (less positive) with 1d EMA34 downtrend and volume spike
# Volume spike > 1.5x 20-period EMA confirms institutional participation
# Designed for 6h timeframe: targets 12-30 trades/year (50-120 total over 4 years)
# Works in bull/bear: trend filter adapts to market regime, volume avoids false signals

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # 1d EMA34 for trend filter
    df_1d_close = pd.Series(df_1d['close'].values)
    ema34_1d = df_1d_close.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume spike filter: volume > 1.5 * 20-period EMA
    df_1d_volume = pd.Series(df_1d['volume'].values)
    vol_ema_20_1d = df_1d_volume.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    volume_spike = volume > (1.5 * vol_ema_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 34)  # Need EMA13 and 1d EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 slope
        if i >= start_idx + 1:
            ema34_prev = ema34_1d_aligned[i-1]
            ema34_curr = ema34_1d_aligned[i]
            trend_up = ema34_curr > ema34_prev
            trend_down = ema34_curr < ema34_prev
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power positive AND rising (less negative bear power) with uptrend and volume
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear power rising (less negative)
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND falling (less positive bull power) with downtrend and volume
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull power falling (less positive)
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power turns positive (bulls losing control) or trend breaks
            if bear_power[i] >= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (bears losing control) or trend breaks
            if bull_power[i] <= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals