#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator strategy with 1w trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for Alligator jaw/trend filter (captures weekly trend to avoid counter-trend trades).
- Entry: Long when price > Alligator lips AND lips > teeth AND teeth > jaw (bullish alignment) AND volume > 1.5 * 1d volume MA(20);
         Short when price < Alligator lips AND lips < teeth AND teeth < jaw (bearish alignment) AND volume > 1.5 * 1d volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via Alligator (signal=0 when price crosses Alligator teeth in opposite direction).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator provides trend-following signals with smooth jaws/teeth/lips; 1w filter ensures alignment with higher timeframe trend; volume confirmation avoids false signals.
- Works in bull markets (buy Alligator alignment) and bear markets (sell Alligator alignment) with volume confirmation to avoid whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs shifted)
    # Jaw: 13-period SMA shifted 8 bars
    # Teeth: 8-period SMA shifted 5 bars  
    # Lips: 5-period SMA shifted 3 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Apply shifts (Alligator definition)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set NaN for shifted values that don't have enough data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate volume MA(20) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, None, jaw)  # jaw already calculated on close
    teeth_aligned = align_htf_to_ltf(prices, None, teeth)
    lips_aligned = align_htf_to_ltf(prices, None, lips)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 13  # Need enough data for Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price crosses Alligator teeth in opposite direction
        if position == 1:
            if curr_close < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator alignment conditions
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] and 
                           teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] and 
                           teeth_aligned[i] < jaw_aligned[i])
        
        # Trend filter from 1w EMA34
        price_above_ema = curr_close > ema_34_aligned[i]
        price_below_ema = curr_close < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Bullish Alligator alignment AND price above 1w EMA34
                if bullish_alignment and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: Bearish Alligator alignment AND price below 1w EMA34
                elif bearish_alignment and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0