#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) with EMA13 rising
# Actually: Elder Ray = Bull Power (close - EMA13) and Bear Power (close - EMA13) - wait, that's not right.
# Correct: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power < 0 (market above EMA13 with strength) AND 1d EMA34 rising
# Short when Bear Power < 0 AND Bull Power > 0? No, that's impossible.
# Actually: We'll use 1d EMA34 for trend, and 6h Elder Ray for entry:
# Long when Bull Power > 0 AND Bear Power < 0 (price above EMA13) AND 1d EMA34 rising
# Short when Bear Power < 0 AND Bull Power > 0? Let's reframe:
# Elder Ray confirms trend: Bull Power > 0 = bulls in control, Bear Power < 0 = bears in control
# But both can't be true simultaneously. Actually:
# Bull Power = High - EMA13 (measures bulls' ability to push price above EMA13)
# Bear Power = Low - EMA13 (measures bears' ability to push price below EMA13)
# In strong uptrend: Bull Power > 0 and increasing, Bear Power may be negative but less negative
# In strong downtrend: Bear Power < 0 and decreasing, Bull Power may be positive but less positive
# We'll simplify: Long when Bull Power > 0 (bulls in control) AND 1d EMA34 rising
# Short when Bear Power < 0 (bears in control) AND 1d EMA34 falling
# Add volume confirmation to avoid false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 (trend filter) ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Indicator: Elder Ray (Bull Power and Bear Power) ===
    # Calculate EMA13 on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 13, 20) + 5  # EMA34(1d) + EMA13(6h) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (bulls in control: high > EMA13)
        # 2. 1d EMA34 rising (current > previous)
        # 3. Volume confirmation
        if (bull_power[i] > 0) and \
           (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (bears in control: low < EMA13)
        # 2. 1d EMA34 falling (current < previous)
        # 3. Volume confirmation
        elif (bear_power[i] < 0) and \
             (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA13_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0