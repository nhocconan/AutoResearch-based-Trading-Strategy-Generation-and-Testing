# Your Name: 
# Your Unique Idea: 
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike with 1d trend filter.
# Williams Alligator identifies trend direction (Jaw/Teeth/Lips alignment).
# Elder Ray measures bull/bear power via EMA13.
# Volume spike confirms breakout strength.
# 1d trend filter ensures alignment with higher timeframe momentum.
# Designed for ~20-30 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price
    # Jaw (13-period, shifted 8), Teeth (8-period, shifted 5), Lips (5-period, shifted 3)
    median_price_1d = (high_1d + low_1d) / 2.0
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align 1d indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need enough data for Alligator (13+8=21 bars) + volume MA
    start_idx = 25
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Williams Alligator: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power > 0 indicate strength
        bullish_elder = bull_power_aligned[i] > 0
        bearish_elder = bear_power_aligned[i] > 0
        
        if position == 0:
            # Long: Alligator aligned up + Elder Ray bullish + volume spike
            if bullish_alligator and bullish_elder and vol_filter:
                signals[i] = size
                position = 1
            # Short: Alligator aligned down + Elder Ray bearish + volume spike
            elif bearish_alligator and bearish_elder and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks or Elder Ray turns bearish
            if not bullish_alligator or not bullish_elder:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator alignment breaks or Elder Ray turns bullish
            if not bearish_alligator or not bearish_elder:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0