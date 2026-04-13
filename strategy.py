#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator trend filter + 1w/1d HTF regime + volume confirmation
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs
    # Trend up when Lips > Teeth > Jaw, trend down when Lips < Teeth < Jaw
    # Only trade in direction of 1w EMA50 trend (bull/bear filter)
    # Enter on pullback to Teeth (8-period) in trend direction with volume > 1.5x 20-bar avg
    # Exit when price crosses Jaw (13-period) or Alligator lines intertwine (chop regime)
    # Uses 1w HTF for major trend filter to avoid counter-trend trades in strong regimes
    # Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
    # Position size: 0.25 (25%) to balance risk and return
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator SMAs (12h timeframe)
    # Jaw: 13-period SMMA, smoothed with 8-period
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean()
    jaw_12h = jaw_12h.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMMA, smoothed with 5-period
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean()
    teeth_12h = teeth_12h.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMMA, smoothed with 3-period
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean()
    lips_12h = lips_12h.rolling(window=3, min_periods=3).mean().values
    
    # Align 12h Alligator lines to 12h timeframe (no-op but for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1w data for major trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for major trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for volume confirmation and chop filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d volume confirmation: volume > 1.5x 20-bar average
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (1.5 * avg_volume_1d)
    volume_confirmed = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d)
    
    # 1d chop filter: choppiness index > 61.8 = ranging market (avoid trend following in chop)
    # Calculate True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) for 1d
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14_1d - ll_14_1d
    chop_1d = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop_1d[range_14 == 0] = 100  # avoid division by zero
    
    # Chop filter: only trade when market is trending (CHOP < 61.8)
    chop_filter = chop_1d < 61.8
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(13, n):  # start after Alligator warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirmed[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend conditions
        # Trend up: Lips > Teeth > Jaw
        trend_up = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Trend down: Lips < Teeth < Jaw
        trend_down = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Alligator intertwining (chop regime) - lines are close together
        jaw_teeth_diff = np.abs(jaw_aligned[i] - teeth_aligned[i])
        teeth_lips_diff = np.abs(teeth_aligned[i] - lips_aligned[i])
        avg_price = (close[i] + close[i-1]) / 2 if i > 0 else close[i]
        intertwining = (jaw_teeth_diff < 0.01 * avg_price) and (teeth_lips_diff < 0.01 * avg_price)
        
        # Entry conditions: pullback to Teeth in trend direction
        # Long: price pulls back to or slightly below Teeth in uptrend
        pullback_to_teeth_long = close[i] <= teeth_aligned[i] * 1.002  # within 0.2% above Teeth
        # Short: price pulls back to or slightly above Teeth in downtrend
        pullback_to_teeth_short = close[i] >= teeth_aligned[i] * 0.998  # within 0.2% below Teeth
        
        long_entry = (trend_up and pullback_to_teeth_long and 
                     volume_confirmed[i] and chop_filter_aligned[i] and 
                     close[i] > ema_50_1w_aligned[i] and position != 1)
        short_entry = (trend_down and pullback_to_teeth_short and 
                      volume_confirmed[i] and chop_filter_aligned[i] and 
                      close[i] < ema_50_1w_aligned[i] and position != -1)
        
        # Exit conditions
        # Exit when price crosses Jaw (trend weakening) or Alligator intertwines (chop)
        exit_long = (position == 1 and (close[i] < jaw_aligned[i] or intertwining))
        exit_short = (position == -1 and (close[i] > jaw_aligned[i] or intertwining))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_williams_alligator_trend_volume_chop_filter_v1"
timeframe = "12h"
leverage = 1.0