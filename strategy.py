#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# - Williams Alligator (Jaw=TEETH=LIPS) identifies trendless markets (all lines intertwined)
# - Elder Ray measures bull/bear power via EMA(13) relative to high/low
# - Long when: Alligator lines separated upward (JAW < TEETH < LIPS) AND Bull Power > 0 AND Bear Power < 0
# - Short when: Alligator lines separated downward (JAW > TEETH > LIPS) AND Bear Power < 0 AND Bull Power < 0
# - Uses 1d EMA(50) as regime filter: only long when price > 1d EMA50, short when price < 1d EMA50
# - Alligator period defaults: 13,8,5 smoothed with 8,5,3 respectively
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: Alligator detects trend, Elder Ray measures strength

name = "6h_1d_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for regime filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Pre-compute Williams Alligator on 6h data
    # Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA (smoothed moving average)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    def smma(arr, period):
        """Smoothed Moving Average - similar to EMA but with alpha=1/period"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Smoothed with offset (as per original Alligator)
    jaw = np.roll(jaw, 8)   # Jaw smoothed by 8 bars
    teeth = np.roll(teeth, 5) # Teeth smoothed by 5 bars
    lips = np.roll(lips, 3)   # Lips smoothed by 3 bars
    
    # Pre-compute Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        alligator_bull = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])  # Jaw < Teeth < Lips = bullish alignment
        alligator_bear = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])  # Jaw > Teeth > Lips = bearish alignment
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0  # Bull power positive
        bear_weak = bear_power[i] < 0    # Bear power negative
        
        # Regime filter from 1d EMA50
        price_above_1d_ema = prices['close'].iloc[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = prices['close'].iloc[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish AND Bull power > 0 AND Bear power < 0 AND price > 1d EMA50
            if (alligator_bull and bull_strong and bear_weak and price_above_1d_ema):
                position = 1
                signals[i] = 0.25
            # Short: Alligator bearish AND Bear power < 0 AND Bull power < 0 AND price < 1d EMA50
            elif (alligator_bear and bear_weak and bull_strong and price_below_1d_ema):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when Alligator lines re-intertwine (trend weakness) OR power signals fade
            alligator_flat = not (alligator_bull or alligator_bear)  # Lines intertwined
            power_fade = (bull_power[i] <= 0) or (bear_power[i] >= 0)  # Power signals fade
            
            if alligator_flat or power_fade:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals