#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with weekly trend filter and volume confirmation.
# Long when green line > red line (bullish alignment) AND price above weekly EMA50 AND volume spike (>1.5x average).
# Short when green line < red line (bearish alignment) AND price below weekly EMA50 AND volume spike.
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend alignment, weekly EMA50 for higher timeframe trend filter.
# Volume confirmation ensures moves have institutional participation. Target: 12-37 trades/year per symbol.
name = "12h_WilliamsAlligator_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams Alligator calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator lines (Jaw=13, Teeth=8, Lips=5 SMAs of median price)
    median_price_1w = (high_1w + low_1w) / 2
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values  # Jaw (blue)
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values    # Teeth (red)
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values     # Lips (green)
    
    # Align Alligator lines to 12h timeframe (wait for weekly close)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Get daily data for weekly EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA50 on daily close (proxy for weekly trend)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 12h timeframe (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Need volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ma = vol_ma_30[i]
        vol = volume[i]
        
        # Williams Alligator signals: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: bullish Alligator alignment AND price above weekly EMA50 AND volume confirmation
            if bullish_alignment and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator alignment AND price below weekly EMA50 AND volume confirmation
            elif bearish_alignment and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Alligator alignment turns bearish or price below weekly EMA50
            if not bullish_alignment or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Alligator alignment turns bullish or price above weekly EMA50
            if not bearish_alignment or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals