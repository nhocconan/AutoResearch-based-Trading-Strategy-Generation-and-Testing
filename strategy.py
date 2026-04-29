#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with Daily Volume Spike and Choppiness Filter
# Camarilla pivot levels (R1, S1) act as intraday support/resistance in ranging markets
# Breakouts from these levels with volume confirmation capture momentum moves
# Daily choppiness filter (CHOP > 61.8) ensures we only trade breakouts in ranging markets
# Works in both bull and bear markets by trading mean reversion breaks in ranges
# Target: 20-40 trades/year (80-160 total over 4 years)

name = "4h_Camarilla_R1S1_Breakout_1dChop_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (1.1 * (high_1d - low_1d) / 12.0)
    s1 = close_1d - (1.1 * (high_1d - low_1d) / 12.0)
    
    # Align daily Camarilla levels to 4h timeframe (completed 1d bar only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate daily choppiness index (CHOP) - ranging market filter
    # CHOP > 61.8 = ranging market (good for mean reversion breaks)
    # CHOP < 38.2 = trending market (avoid)
    atr_1d = np.zeros_like(close_1d)
    atr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                 abs(high_1d[i] - close_1d[i-1]),
                 abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = 0.9 * atr_1d[i-1] + 0.1 * tr
    
    # True range sum over 14 periods
    tr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Chop = 100 * log10(tr_sum / range_14) / log10(14)
    chop = np.where(range_14 > 0, 100 * np.log10(tr_sum / range_14) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_chop = chop_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine market regime: CHOP > 61.8 = ranging (trade mean reversion breaks)
        ranging_market = curr_chop > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Only trade in ranging markets
            if ranging_market and curr_volume_confirm:
                # Bullish breakout: price breaks above R1 with volume
                if curr_close > curr_r1:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S1 with volume
                elif curr_close < curr_s1:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to pivot level OR opposite breakout occurs
            if curr_close <= pivot_aligned[i] or (curr_close < curr_s1 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to pivot level OR opposite breakout occurs
            if curr_close >= pivot_aligned[i] or (curr_close > curr_r1 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals