#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Elder Ray + volume spike
# Williams Alligator (jaw/teeth/lips) identifies trend direction and strength
# Elder Ray (bull/bear power) measures buying/selling pressure relative to EMA13
# Volume spike confirms institutional participation
# Designed for low trade frequency: ~12-30 trades/year on 12h timeframe
# Works in bull markets via Alligator uptrend + positive Elder Ray
# Works in bear markets via Alligator downtrend + negative Elder Ray
# Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_WilliamsAlligator_1dElderRay_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
    median_price_12h = (df_12h['high'] + df_12h['low']) / 2
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Alligator trend: bullish when lips > teeth > jaw, bearish when lips < teeth < jaw
    bullish_alligator = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    bearish_alligator = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # 1d HTF data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Elder Ray signals: bullish when bull_power > 0, bearish when bear_power < 0
    bullish_elder = bull_power_aligned > 0
    bearish_elder = bear_power_aligned < 0
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 12h Alligator (21 bars) + 1d Elder Ray (20 bars)
    start_idx = 35
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            if bullish_alligator[i] and bullish_elder[i]:
                # Long: Alligator uptrend + positive Elder Ray + volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_alligator[i] and bearish_elder[i]:
                # Short: Alligator downtrend + negative Elder Ray + volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop or conflicting signals
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Elder Ray turns negative
            if not bullish_alligator[i] or not bullish_elder[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Elder Ray turns positive
            if not bearish_alligator[i] or not bearish_elder[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals