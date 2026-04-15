#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with volume confirmation and 1d ADX trend filter
# Uses Bollinger Bands to identify low volatility squeezes, breaks out in direction of 1d ADX trend,
# confirmed by volume spike. Works in bull/bear by only taking breakouts in ADX trend direction.
# Target: 50-150 total trades over 4 years (12-38/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Bollinger Bands and price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2.0) on 4h
    sma20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma20_4h + 2 * std20_4h
    lower_bb_4h = sma20_4h - 2 * std20_4h
    bb_width_4h = (upper_bb_4h - lower_bb_4h) / sma20_4h
    
    # Bollinger Squeeze: BB width below 20-period mean
    bb_width_ma_20 = pd.Series(bb_width_4h).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width_4h < bb_width_ma_20
    
    # Calculate ADX (14-period) on 1d for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.where(high_1d - np.roll(high_1d, 1) > np.roll(low_1d, 1) - low_1d, 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    down_move = np.where(np.roll(low_1d, 1) - low_1d > high_1d - np.roll(high_1d, 1), 
                         np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smoothed DM
    plus_dm = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm / (atr_1d + 1e-10)
    minus_di = 100 * minus_dm / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume average (20-period on 4h)
    vol_avg_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    upper_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_bb_4h)
    lower_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_bb_4h)
    squeeze_aligned = align_htf_to_ltf(prices, df_4h, squeeze)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_4h_aligned[i]) or np.isnan(lower_bb_4h_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Bollinger squeeze breakout up + volume spike + ADX > 25 (trending) + DI+ > DI-
        if (squeeze_aligned[i] and
            close[i] > upper_bb_4h_aligned[i] and
            volume[i] > 2.0 * vol_avg_aligned[i] and
            adx_aligned[i] > 25 and
            plus_di_aligned[i] > minus_di_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bollinger squeeze breakout down + volume spike + ADX > 25 (trending) + DI- > DI+
        elif (squeeze_aligned[i] and
              close[i] < lower_bb_4h_aligned[i] and
              volume[i] > 2.0 * vol_avg_aligned[i] and
              adx_aligned[i] > 25 and
              minus_di_aligned[i] > plus_di_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or ADX < 20 (losing trend)
        elif position == 1 and (close[i] < lower_bb_4h_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper_bb_4h_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Squeeze_Volume_ADX"
timeframe = "4h"
leverage = 1.0