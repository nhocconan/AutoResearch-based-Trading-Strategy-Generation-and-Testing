#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume confirmation (>1.8x 20-period MA).
Long when price breaks above R1 in 1w uptrend with volume spike. Short when price breaks below S1 in 1w downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. Camarilla levels calculated from prior 1d OHLC to avoid look-ahead.
Designed to work in both bull and bear markets by following the 1w trend. Target: 7-25 trades/year (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC (avoid look-ahead)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Use prior day's OHLC (shifted by 1)
    prior_high = pd.Series(high).shift(1).values
    prior_low = pd.Series(low).shift(1).values
    prior_close = pd.Series(close).shift(1).values
    camarilla_upper = prior_close + (prior_high - prior_low) * 1.1 / 12
    camarilla_lower = prior_close - (prior_high - prior_low) * 1.1 / 12
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    uptrend_1w = close > ema_34_1w_aligned
    downtrend_1w = close < ema_34_1w_aligned
    
    # Volume confirmation: volume > 1.8x 20-period MA (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 1 for shift + 20 for volume MA + 34 for 1w EMA)
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_upper[i]) or np.isnan(camarilla_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above camarilla_upper with 1w uptrend and volume spike
            if (close[i] > camarilla_upper[i] and 
                uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below camarilla_lower with 1w downtrend and volume spike
            elif (close[i] < camarilla_lower[i] and 
                  downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below camarilla_lower (breakdown) OR 1w trend changes to downtrend
            if (close[i] < camarilla_lower[i] or not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above camarilla_upper (breakout) OR 1w trend changes to uptrend
            if (close[i] > camarilla_upper[i] or not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0