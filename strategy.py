#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + 1w trend filter
# - Williams Alligator: Jaw (EMA13, 8 bars offset), Teeth (EMA8, 5 bars offset), Lips (EMA5, 3 bars offset)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1d volume > 1.5x 20-period volume SMA AND 1w close > 1w EMA50
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1d volume > 1.5x 20-period volume SMA AND 1w close < 1w EMA50
# - Exit: Alligator lines cross (Lips crosses Teeth) or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Alligator for trend identification, volume for confirmation, 1w EMA50 for higher timeframe trend filter

name = "12h_1d_1w_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Williams Alligator
    # Jaw: EMA13 of median price, offset 8 bars
    # Teeth: EMA8 of median price, offset 5 bars  
    # Lips: EMA5 of median price, offset 3 bars
    median_price = (high + low) / 2.0
    
    # Calculate EMAs
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply offsets (shift right to avoid look-ahead)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set invalid values for offset periods
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (already on 12h, but ensuring proper alignment)
    jaw_aligned = jaw  # No additional alignment needed as we're on 12h
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(60, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_12h = volume[i] > 1.5 * volume_sma_20_1d_aligned[i]  # Using 1d volume SMA as proxy for 12h
        vol_confirm_1d = volume_1d[i] > 1.5 * volume_sma_20_1d_aligned[i] if i < len(volume_1d) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Alligator alignment signals
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Exit conditions: Alligator lines cross (Lips crosses Teeth) or loss of volume confirmation
        exit_long = lips_aligned[i] <= teeth_aligned[i] or not vol_confirm
        exit_short = lips_aligned[i] >= teeth_aligned[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if bullish_alignment and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif bearish_alignment and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals