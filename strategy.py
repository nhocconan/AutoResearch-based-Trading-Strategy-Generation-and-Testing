#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams Alligator + Daily Trend Filter + Volume Spike
# Hypothesis: Williams Alligator identifies trend phases with clear entry/exit rules.
# Daily trend filter (EMA50) ensures alignment with higher-timeframe momentum.
# Volume spike confirms institutional participation in the trend.
# Designed for 6h timeframe with low trade frequency (12-37/year).
# Works in bull via long signals when jaws-teeth-lips aligned up + daily uptrend + volume,
# in bear via short signals when jaws-teeth-lips aligned down + daily downtrend + volume.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_williams_alligator_1d_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5) - Smoothed Moving Average (SMMA)
    def smma(values, period):
        """Smoothed Moving Average - Williams Alligator uses SMMA"""
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Alligator lines: Jaw(13,8), Teeth(8,5), Lips(5,3)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Daily trend filter: EMA(50) of daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Alligator lines cross in wrong order OR daily trend turns bearish
            if not (jaw[i] > teeth[i] > lips[i]) or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Alligator lines cross in wrong order OR daily trend turns bullish
            if not (jaw[i] < teeth[i] < lips[i]) or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: Alligator aligned up (Jaw > Teeth > Lips) + price above daily EMA50
                if jaw[i] > teeth[i] > lips[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: Alligator aligned down (Jaw < Teeth < Lips) + price below daily EMA50
                elif jaw[i] < teeth[i] < lips[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals