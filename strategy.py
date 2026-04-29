#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator uses smoothed medians (Jaw=13, Teeth=8, Lips=5) to identify trends.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when Alligator alignment breaks (Lips crosses Teeth) or opposite signal occurs.
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-30 trades/year on 6h.
# Alligator filters choppy markets, EMA34 ensures higher timeframe trend alignment,
# Volume confirmation avoids false breakouts. Works in both bull and bear markets by following trends.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data
    # Median price = (high + low + close) / 3
    median_price = (high + low + close) / 3.0
    
    # Jaw: Blue line - 13-period SMMA shifted 8 bars ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: Red line - 8-period SMMA shifted 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: Green line - 5-period SMMA shifted 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # EMA34 needs 34 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Check Alligator alignment
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when bullish alignment AND price > 1d EMA34 AND volume confirmation
            if bullish_alignment and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND price < 1d EMA34 AND volume confirmation
            elif bearish_alignment and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when alignment breaks or opposite signal
            if not bullish_alignment:  # Exit when Alligator alignment breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when alignment breaks or opposite signal
            if not bearish_alignment:  # Exit when Alligator alignment breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals