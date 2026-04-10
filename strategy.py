#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w trend filter + volume confirmation
# - Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA20 AND 1d volume > 1.5x 20-period volume SMA
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA20 AND 1d volume > 1.5x 20-period volume SMA
# - Exit: Alligator lines cross (Lips crosses Teeth) or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 20-50 trades/year on 1d timeframe to stay within fee drag limits
# - Williams Alligator catches trends early with built-in smoothing; 1w filter avoids counter-trend trades; volume confirms conviction

name = "1d_williams_alligator_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 1d timeframe (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Self-align for same timeframe
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 1d volume SMA for confirmation
    volume_sma_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(volume_sma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20_1d[i]
        
        # Trend filter: 1w close vs 1w EMA20
        trend_bullish = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Williams Alligator signals
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