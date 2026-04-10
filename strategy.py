#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume spike regime filter
# - Uses 12h timeframe to reduce trade frequency and fee drag
# - Williams Alligator (jaw/teeth/lips) for trend direction and strength
# - Elder Ray (bull/bear power) for momentum confirmation
# - Volume spike (>2x 20-period average) for conviction
# - Long when: Alligator aligned bullish (lips>teeth>jaw) AND Elder Bull Power > 0 AND volume spike
# - Short when: Alligator aligned bearish (lips<teeth<jaw) AND Elder Bear Power < 0 AND volume spike
# - ATR-based trailing stop: exit when price moves 2.5x ATR against position
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Works in both bull/bear markets: Alligator adapts to trend, Elder Ray measures momentum strength

name = "12h_williams_alligator_elder_ray_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Williams Alligator calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Williams Alligator (SMAs of median price)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_price_1d = (high_1d + low_1d) / 2
    
    # Williams Alligator: jaw=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3)
    # Using median price as input
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray to 12h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 12h ATR(14) for stoploss calculation
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_12h = np.zeros_like(tr)
    atr_14_12h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_12h[i] = (atr_14_12h[i-1] * (14-1) + tr[i]) / 14
    
    # 12h volume confirmation: > 2.0x 20-period average
    volume_12h = prices['volume'].values
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (2.0 * avg_volume_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(lips_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(atr_14_12h[i]) or np.isnan(vol_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based trailing stop (2.5x ATR)
            if prices['close'].iloc[i] < entry_price - 2.5 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based trailing stop (2.5x ATR)
            if prices['close'].iloc[i] > entry_price + 2.5 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment + Elder Ray + volume spike
            # Bullish alignment: lips > teeth > jaw
            # Bearish alignment: lips < teeth < jaw
            if vol_spike_12h[i]:
                # Long signal: bullish Alligator + positive Bull Power
                if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and
                    bull_power_1d_aligned[i] > 0):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = 0.25
                # Short signal: bearish Alligator + negative Bear Power
                elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and
                      bear_power_1d_aligned[i] < 0):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = -0.25
    
    return signals