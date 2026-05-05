#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when price > Alligator Jaw (TEETH) AND Jaw > Teeth > Lips (bullish alignment) AND close > 1w EMA50 AND volume > 2.0x 20-period average
# Short when price < Alligator Jaw (TEETH) AND Jaw < Teeth < Lips (bearish alignment) AND close < 1w EMA50 AND volume > 2.0x 20-period average
# Exit when Alligator alignment breaks (Jaw-Teeth-Lips not in proper order) OR close crosses 1w EMA50
# Uses 6h primary timeframe with 1w HTF for trend filter to reduce whipsaw and capture strong trends
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag
# Williams Alligator identifies trend via smoothed medians; 1w EMA50 filters for higher timeframe trend; volume confirms breakout strength

name = "6h_Williams_Alligator_1wEMA50_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Williams Alligator (based on daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on daily timeframe
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Jaw (Blue line): 13-period SMMA, shifted 8 bars forward
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw_1d = np.roll(jaw_1d, 8)  # shift forward 8 bars
    jaw_1d[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth (Red line): 8-period SMMA, shifted 5 bars forward
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = np.roll(teeth_1d, 5)  # shift forward 5 bars
    teeth_1d[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips (Green line): 5-period SMMA, shifted 3 bars forward
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips_1d = np.roll(lips_1d, 3)  # shift forward 3 bars
    lips_1d[:3] = np.nan  # first 3 values invalid after shift
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND Jaw > Teeth > Lips (bullish alignment) AND close > 1w EMA50 AND volume spike
            if (close[i] > jaw_aligned[i] and 
                jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND Jaw < Teeth < Lips (bearish alignment) AND close < 1w EMA50 AND volume spike
            elif (close[i] < jaw_aligned[i] and 
                  jaw_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Jaw-Teeth-Lips not bullish) OR close < 1w EMA50 (trend flip)
            if not (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Jaw-Teeth-Lips not bearish) OR close > 1w EMA50 (trend flip)
            if not (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals