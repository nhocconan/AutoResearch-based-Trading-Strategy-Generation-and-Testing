#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
# - Long when Lips cross above Teeth AND Teeth above Jaw (bullish alignment) AND price > Alligator's Mouth (highest of Jaw/Teeth/Lips) AND volume > 1.5x 20-bar avg
# - Short when Lips cross below Teeth AND Teeth below Jaw (bearish alignment) AND price < Alligator's Mouth AND volume > 1.5x 20-bar avg
# - Exit when Lips re-cross Teeth in opposite direction (trend weakening)
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades in strong trends
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years)
# - Williams Alligator excels in trending markets; volume confirmation filters false breakouts

name = "4h_1d_williams_alligator_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator calculation (requires sufficient data for SMMA)
    # SMMA (Smoothed Moving Average) calculation
    def smma(source, length):
        if length < 1:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < length:
            return result
        # First value is simple SMA
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CURRENT_VALUE) / LENGTH
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    close = prices['close'].values
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts: Jaw shifted by 8 bars, Teeth by 5 bars, Lips by 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate the shifted values that now contain rolled data from wrong positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Alligator's Mouth: highest/lowest of the three lines
    # For long: we need price > highest of the three (mouth open upward)
    # For short: we need price < lowest of the three (mouth open downward)
    jaw_teeth_lips = np.column_stack([jaw_shifted, teeth_shifted, lips_shifted])
    alligator_high = np.nanmax(jaw_teeth_lips, axis=1)
    alligator_low = np.nanmin(jaw_teeth_lips, axis=1)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw AND price > Alligator's highest point (mouth open up)
            bullish_align = (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i])
            price_above_mouth = prices['close'].iloc[i] > alligator_high[i]
            
            # Bearish alignment: Lips < Teeth < Jaw AND price < Alligator's lowest point (mouth open down)
            bearish_align = (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i])
            price_below_mouth = prices['close'].iloc[i] < alligator_low[i]
            
            if bullish_align and price_above_mouth and vol_spike.iloc[i]:
                # Additional 1d trend filter: price above 1d EMA50 for long
                if prices['close'].iloc[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
            elif bearish_align and price_below_mouth and vol_spike.iloc[i]:
                # Additional 1d trend filter: price below 1d EMA50 for short
                if prices['close'].iloc[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Lips re-cross Teeth
            # Exit when Lips cross Teeth in opposite direction (trend weakening)
            exit_signal = False
            if position == 1:  # Long position - exit if Lips cross below Teeth
                if lips_shifted[i] < teeth_shifted[i]:
                    exit_signal = True
            elif position == -1:  # Short position - exit if Lips cross above Teeth
                if lips_shifted[i] > teeth_shifted[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals