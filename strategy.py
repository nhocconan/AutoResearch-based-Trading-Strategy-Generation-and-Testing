#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price retouches Camarilla pivot point (PP) or opposite breakout occurs
# Uses discrete position sizing (0.30) to balance profit and fee drag. Target: 20-30 trades/year on 4h.
# Camarilla levels from daily timeframe provide high-probability intraday reversal/continuation points.
# 1d EMA34 filter ensures alignment with daily trend, improving win rate in both bull and bear markets.
# Volume spike confirmation ensures breakouts have institutional conviction, reducing false signals.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    # Use shift(1) to ensure we only use completed daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior completed 1d bar values (avoid look-ahead)
    H1 = pd.Series(high_1d).shift(1).values
    L1 = pd.Series(low_1d).shift(1).values
    C1 = pd.Series(close_1d).shift(1).values
    
    # Calculate Camarilla levels for current 4h bar based on prior 1d bar
    range_1d = H1 - L1
    camarilla_pp = (H1 + L1 + C1 * 2) / 4  # Pivot Point
    camarilla_r3 = camarilla_pp + range_1d * 1.1 / 4  # Resistance 3
    camarilla_s3 = camarilla_pp - range_1d * 1.1 / 4  # Support 3
    
    # Align Camarilla levels to 4h timeframe (already aligned via shift(1) above)
    # No additional alignment needed as we're using prior 1d bar values
    
    # Calculate EMA(34) on 1d close data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe using prior completed 1d bar
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA needs 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        pp = camarilla_pp[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume confirmation
            if curr_high > r3 and curr_close > ema_34 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume confirmation
            elif curr_low < s3 and curr_close < ema_34 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches Camarilla PP or breaks below Camarilla S3
            if curr_close <= pp or curr_low < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - exit when price retouches Camarilla PP or breaks above Camarilla R3
            if curr_close >= pp or curr_high > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals