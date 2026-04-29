#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when Alligator jaws (13-period SMMA) < teeth (8-period SMMA) < lips (5-period SMMA) AND price > 1d EMA34 AND volume > 1.5x 20-bar avg
# Short when Alligator jaws > teeth > lips AND price < 1d EMA34 AND volume > 1.5x 20-bar avg
# Exit when Alligator lines re-interlace (jaws crosses teeth or lips) or price retouches 1d EMA34
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Williams Alligator identifies trending vs ranging markets with clear entry/exit signals.
# 1d EMA34 filter ensures we only trade with the long-term trend, improving win rate in both bull/bear.
# Volume confirmation ensures breakouts have conviction, reducing false signals.

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h data using SMMA (Smoothed Moving Average)
    # SMMA is similar to EMA but with different smoothing - we'll use EMA as approximation
    # Jaws: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    # Calculate SMMA using EMA with adjusted alpha (SMMA ≈ EMA with period*2-1)
    jaws_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply the Alligator shifts (jaw: 8, teeth: 5, lips: 3)
    jaws = np.roll(jaws_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set NaN for shifted values that don't have enough history
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA and Alligator need sufficient bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        jaw = jaws[i]
        tooth = teeth[i]
        lip = lips[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Alligator alignment checks
        bullish_alignment = jaw < tooth < lip  # Jaws < Teeth < Lips (bullish)
        bearish_alignment = jaw > tooth > lip  # Jaws > Teeth > Lips (bearish)
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Alligator bullish AND price > 1d EMA34 AND volume confirmation
            if bullish_alignment and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator bearish AND price < 1d EMA34 AND volume confirmation
            elif bearish_alignment and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Alligator re-interlaces or price retouches 1d EMA34
            if not bullish_alignment or curr_close <= ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Alligator re-interlaces or price retouches 1d EMA34
            if not bearish_alignment or curr_close >= ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals