#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 AND 1w EMA50 is rising AND volume > 1.5x 20-period average.
# Short when price breaks below S1 AND 1w EMA50 is falling AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Camarilla levels provide precise intraday pivot points,
# 1w EMA50 ensures we trade with the higher timeframe trend, volume spike confirms institutional participation.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets with clear structure.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while capturing meaningful moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Typical Price for Camarilla ===
    typical_price = (high + low + close) / 3.0
    
    # Get 1d data once before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Pivot point = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_values = ema_50_1w.values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_values)
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50, 20 for volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        tp = typical_price[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_50_1w = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point or volume spike ends
            if tp <= pp_aligned[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point or volume spike ends
            if tp >= pp_aligned[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND 1w EMA50 is rising (today > yesterday) AND volume spike
            if tp > r1 and ema_50_1w > ema_50_1w_aligned[max(i-1, warmup)] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 AND 1w EMA50 is falling (today < yesterday) AND volume spike
            elif tp < s1 and ema_50_1w < ema_50_1w_aligned[max(i-1, warmup)] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wEMA50_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0