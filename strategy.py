#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Elder Ray + volume confirmation
# - Uses 12h Alligator (JAW=13, TEETH=8, LIPS=5) for trend direction
# - Confirms with 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
# - Requires 1d volume > 1.5x 20-period average for institutional participation
# - Enters long when Alligator bullish (LIPS > TEETH > JAW) AND Elder Ray bullish (Bull Power > 0)
# - Enters short when Alligator bearish (LIPS < TEETH < JAW) AND Elder Ray bearish (Bear Power > 0)
# - Exits when Alligator reverses or volume drops below average
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to minimize fee drag
# - Works in bull markets (Alligator alignment + Elder Ray confirmation) and bear markets (same logic for shorts)
# - Williams Alligator identifies trending vs ranging markets; Elder Ray measures bull/bear strength

name = "12h_1d_alligator_elder_ray_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(13) for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # 1d Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.5 * avg_volume_20)
    
    # Load 12h data for Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Williams Alligator: SMAs of median price
    median_price_12h = (high_12h + low_12h) / 2.0
    
    # JAW: 13-period SMA, shifted by 8 bars
    jaw_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values
    jaw_12h = np.roll(jaw_12h, 8)
    jaw_12h[:8] = np.nan
    
    # TEETH: 8-period SMA, shifted by 5 bars
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)
    teeth_12h[:5] = np.nan
    
    # LIPS: 5-period SMA, shifted by 3 bars
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)
    lips_12h[:3] = np.nan
    
    # Align all 1d indicators to 12h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    
    # Align Alligator components to 12h (already 12h, but ensure alignment)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            jaw_aligned[i] <= 0 or teeth_aligned[i] <= 0 or lips_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator turns bearish OR volume confirmation lost
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or not volume_confirm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR volume confirmation lost
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or not volume_confirm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Alligator bullish AND Elder Ray bullish AND volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and      # Alligator bullish alignment
                bull_power_aligned[i] > 0 and                              # Elder Ray bullish
                volume_confirm_aligned[i]):                                # Volume confirmation
                position = 1
                signals[i] = 0.25
            # Enter short: Alligator bearish AND Elder Ray bearish AND volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and  # Alligator bearish alignment
                  bear_power_aligned[i] > 0 and                            # Elder Ray bearish
                  volume_confirm_aligned[i]):                              # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals