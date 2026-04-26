#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA34 trend filter and volume spike confirmation.
- Long when price breaks above daily Camarilla R1 AND weekly EMA34 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below daily Camarilla S1 AND weekly EMA34 downtrend AND volume > 1.5 * volume_ma(20)
- Uses Camarilla levels from daily chart for structure-based breakouts (R1/S1 = inner support/resistance)
- Weekly EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws in bear markets
- Volume spike confirms institutional participation and reduces false breakouts
- Designed for low frequency (target 7-25 trades/year on 1d) to minimize fee drag
- Exit on opposite Camarilla level touch (R1/S1) or trend reversal
- Novelty: Uses tighter Camarilla levels (R1/S1 vs R3/S3) for earlier entry with trend/volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior daily bar (completed bar only)
    # Camarilla formulas based on previous day's range
    # R1 = close + 1.0833*(high-low), S1 = close - 1.0833*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data only
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + 1.0833 * range_1d
    camarilla_s1 = prev_close - 1.0833 * range_1d
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed for structure)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter (needs completed weekly candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1w = np.where(ema_34_1w_aligned > 0, 
                        np.where(close > ema_34_1w_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(trend_1w[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R1/S1 breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND weekly uptrend AND volume spike
            if close[i] > camarilla_r1_aligned[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND weekly downtrend AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR weekly trend turns down
            if close[i] < camarilla_s1_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR weekly trend turns up
            if close[i] > camarilla_r1_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0