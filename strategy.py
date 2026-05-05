#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period)
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1d EMA34 AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when Alligator alignment breaks (Lips crosses Teeth) OR close crosses 1d EMA34
# Uses 6h primary timeframe with 1d HTF for trend filter to capture multi-day moves with controlled frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Williams Alligator identifies trending vs ranging markets; EMA34 filters for higher-timeframe trend; volume confirms participation

name = "6h_Williams_Alligator_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components on 6h data
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period smoothed median, 8-period offset
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values  # additional smoothing
    
    # Teeth: 8-period smoothed median, 5-period offset
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values  # additional smoothing
    
    # Lips: 5-period smoothed median, 3-period offset
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values  # additional smoothing
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long conditions: bullish Alligator alignment AND close > 1d EMA34 AND volume spike
            if bullish_alignment and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Alligator alignment AND close < 1d EMA34 AND volume spike
            elif bearish_alignment and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR close < 1d EMA34 (trend flip)
            if not bullish_alignment or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR close > 1d EMA34 (trend flip)
            if not bearish_alignment or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals