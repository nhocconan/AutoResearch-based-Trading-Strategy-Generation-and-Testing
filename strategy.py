#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w EMA50 for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Williams Alligator: Jaw (13-period SMA smoothed 8 bars), Teeth (8-period SMA smoothed 5 bars), Lips (5-period SMA smoothed 3 bars).
- Volume: Current 1d volume > 1.5 * 20-period volume MA to avoid low-volume signals.
- Entry: Long when Lips > Teeth > Jaw (Alligator bullish) AND 1w EMA50 bullish AND volume spike.
         Short when Lips < Teeth < Jaw (Alligator bearish) AND 1w EMA50 bearish AND volume spike.
- Exit: Opposite Alligator alignment (Lips crosses Teeth or Jaw) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Williams Alligator identifies trend phases; combined with 1w trend and volume filters, avoids whipsaws and works in both bull and bear markets by only taking trades in the direction of the 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator
    # Jaw: 13-period SMA, smoothed by 8 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 bars
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 bars
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 1w
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 1d volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # Need enough bars for EMA50, volume MA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        ema_val = ema_1w_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Lips > Teeth > Jaw (Alligator bullish) AND 1w EMA50 bullish (price > EMA)
                if lips_val > teeth_val and teeth_val > jaw_val and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips < Teeth < Jaw (Alligator bearish) AND 1w EMA50 bearish (price < EMA)
                elif lips_val < teeth_val and teeth_val < jaw_val and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Lips crosses below Teeth OR loss of volume confirmation
            if lips_val < teeth_val or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips crosses above Teeth OR loss of volume confirmation
            if lips_val > teeth_val or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0