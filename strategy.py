#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA trend filter + volume spike
# Long when Alligator jaws < teeth < lips (bullish alignment) + price > lips + 1d EMA50 up + volume > 2x 20-period avg
# Short when Alligator jaws > teeth > lips (bearish alignment) + price < lips + 1d EMA50 down + volume > 2x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (15-35/year).
# Alligator identifies trend early with smoothed MAs. 1d EMA filter ensures we only trade with higher timeframe trend.
# Works in bull markets (Alligator alignment + rising 1d EMA) and bear markets (reverse alignment + falling 1d EMA).

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
    
    # === 1d Indicator: EMA(50) for trend direction ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicator: Williams Alligator (13,8,5) with smoothing ===
    # Alligator: Jaw (13-period SMMA, 8 bars shift), Teeth (8-period SMMA, 5 bars shift), Lips (5-period SMMA, 3 bars shift)
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - similar to EMA but with alpha=1/period"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN for invalid periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 13+8, 20) + 5  # EMA(50) + Alligator shifts + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: jaw < teeth < lips
        # 2. Price above lips (strong bullish momentum)
        # 3. 1d EMA50 trending up (current > previous)
        # 4. Volume confirmation
        if (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]) and \
           (close[i] > lips_shifted[i]) and \
           (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: jaw > teeth > lips
        # 2. Price below lips (strong bearish momentum)
        # 3. 1d EMA50 trending down (current < previous)
        # 4. Volume confirmation
        elif (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]) and \
             (close[i] < lips_shifted[i]) and \
             (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0