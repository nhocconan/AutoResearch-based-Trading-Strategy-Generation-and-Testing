#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R1 (Camarilla resistance) AND 1d EMA50 uptrend (price > EMA50) AND volume > 1.3x 20-period average.
# Short when price breaks below S1 (Camarilla support) AND 1d EMA50 downtrend (price < EMA50) AND volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Camarilla levels provide institutional support/resistance, 1d EMA50 ensures trend alignment,
# volume spike confirms institutional participation. Designed to work in both bull (breakout longs) and bear (breakdown shorts).
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag while staying within 12h limits.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Points (based on previous bar) ===
    # Calculate pivot and levels using previous bar's OHLC
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # === 12h Indicators: Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r1 = R1[i]
        s1 = S1[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls back below pivot (failed breakout) or volume spike ends
            if price < pivot[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises back above pivot (failed breakdown) or volume spike ends
            if price > pivot[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND price > 1d EMA50 (uptrend) AND volume spike
            if price > r1 and price > ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 AND price < 1d EMA50 (downtrend) AND volume spike
            elif price < s1 and price < ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_CamarillaR1S1_Breakout_1dEMA50_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0