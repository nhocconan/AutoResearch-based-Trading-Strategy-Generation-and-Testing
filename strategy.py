#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrendFilter_VolumeConfirm_v1
Hypothesis: 12h Camarilla breakout with daily trend filter and volume confirmation.
- Uses 12h timeframe for low trade frequency (target: 12-37/year)
- Camarilla pivot levels (R1, S1) from daily data as support/resistance
- Daily EMA34 filter ensures trades align with higher timeframe trend
- Volume spike confirmation (>1.5x 20-period average) reduces false breakouts
- Long when price breaks above R1 with volume spike AND daily uptrend
- Short when price breaks below S1 with volume spike AND daily downtrend
- Exit when price returns to mean (PP) or opposite Camarilla level touched
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the daily trend and using Camarilla for structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels from previous day's data
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_r1 = prev_close + (camarilla_range * 1.1 / 12)
    camarilla_s1 = prev_close - (camarilla_range * 1.1 / 12)
    
    # Align daily Camarilla levels to 12h timeframe (wait for completed daily bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation
        price_above_r1 = close[i] > camarilla_r1_aligned[i]
        price_below_s1 = close[i] < camarilla_s1_aligned[i]
        vol_confirmed = volume_spike[i]
        
        # Daily trend filter
        daily_uptrend = close[i] > ema34_1d_aligned[i]
        daily_downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND daily uptrend
            if price_above_r1 and vol_confirmed and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND daily downtrend
            elif price_below_s1 and vol_confirmed and daily_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price returns to PP (mean reversion) or breaks below S1
            if close[i] <= camarilla_pp_aligned[i] or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price returns to PP (mean reversion) or breaks above R1
            if close[i] >= camarilla_pp_aligned[i] or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0