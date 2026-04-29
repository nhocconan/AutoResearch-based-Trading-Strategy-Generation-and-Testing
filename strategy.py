#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 Breakout with 4h Trend Filter and Volume Spike
# Uses 4h Camarilla pivot levels (R1/S1) as intraday support/resistance
# Breakouts above R1 or below S1 with volume confirmation capture momentum
# 4h EMA50 filter ensures we only trade breakouts in the direction of the 4h trend
# Session filter (08-20 UTC) reduces noise outside active trading hours
# Target: 15-37 trades/year (60-150 total over 4 years)

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (R1, S1) from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = close_4h + (1.1 * (high_4h - low_4h) * 1.0 / 4.0)  # R1 = C + 1.1*(H-L)/4
    s1_4h = close_4h - (1.1 * (high_4h - low_4h) * 1.0 / 4.0)  # S1 = C - 1.1*(H-L)/4
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe (completed 4h bar only)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_ema50 = ema_50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine 4h trend: price above/below EMA50
        uptrend = curr_close > curr_ema50
        downtrend = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in direction of 4h trend
            if curr_volume_confirm:
                # Bullish breakout: price breaks above R1 in uptrend
                if uptrend and curr_close > curr_r1:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below S1 in downtrend
                elif downtrend and curr_close < curr_s1:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to 4h pivot OR breaks below S1 with volume
            high_4h_i = df_4h['high'].values
            low_4h_i = df_4h['low'].values
            close_4h_i = df_4h['close'].values
            pivot_4h_i = (high_4h_i + low_4h_i + close_4h_i) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h_i)
            curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else curr_r1  # fallback
            
            if curr_close <= curr_pivot or (curr_close < curr_s1 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to 4h pivot OR breaks above R1 with volume
            high_4h_i = df_4h['high'].values
            low_4h_i = df_4h['low'].values
            close_4h_i = df_4h['close'].values
            pivot_4h_i = (high_4h_i + low_4h_i + close_4h_i) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h_i)
            curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else curr_s1  # fallback
            
            if curr_close >= curr_pivot or (curr_close > curr_r1 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals