#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses actual Camarilla pivot levels from prior day for precise entry/exit levels.
# Trend filter ensures we trade with higher timeframe momentum.
# Volume confirmation avoids false breakouts.
# Target: 20-40 trades/year (80-160 total over 4 years) for optimal fee drag control.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate prior day's Camarilla levels (using previous day's OHLC)
    # Camarilla equations:
    # P = (H + L + C) / 3
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    day_range = prev_day_high - prev_day_low
    r1 = prev_day_close + day_range * 1.1 / 12
    s1 = prev_day_close - day_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (completed daily bar only)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout above R1 with uptrend and volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema_34 and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakdown below S1 with downtrend and volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema_34 and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when price breaks below S1 (failed breakout) or volume dries up
            if curr_close < curr_s1 or not curr_volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when price breaks above R1 (failed breakdown) or volume dries up
            if curr_close > curr_r1 or not curr_volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals