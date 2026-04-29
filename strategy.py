#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla levels provide precise intraday support/resistance; EMA34 filters trend direction;
# Volume confirms breakout strength. Works in bull/bear by trading breakouts in trend direction.
# Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: 
    # H4 = Close + 1.5*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low)
    # L3 = Close - 1.125*(High-Low)
    # H2 = Close + 0.75*(High-Low)
    # L2 = Close - 0.75*(High-Low)
    # H1 = Close + 0.5*(High-Low)
    # L1 = Close - 0.5*(High-Low)
    # We focus on H3/L3 and H4/L4 for breakouts
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    daily_range = prev_day_high - prev_day_low
    h3 = prev_day_close + 1.125 * daily_range
    l3 = prev_day_close - 1.125 * daily_range
    h4 = prev_day_close + 1.5 * daily_range
    l4 = prev_day_close - 1.5 * daily_range
    
    # Align Camarilla levels to 12h timeframe (completed daily bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 24)  # warmup for EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(ema_34_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_h4 = h4_aligned[i]
        curr_l4 = l4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above H3/H4 with volume spike and uptrend
            if curr_close > curr_h3 and curr_volume_spike and curr_close > curr_ema_34:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below L3/L4 with volume spike and downtrend
            elif curr_close < curr_l3 and curr_volume_spike and curr_close < curr_ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below L3 (reversal) OR trend changes
            if curr_close < curr_l3 or curr_close < curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above H3 (reversal) OR trend changes
            if curr_close > curr_h3 or curr_close > curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals