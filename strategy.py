#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends
# Trades with 1d trend: long when Alligator is bullish (Lips > Teeth > Jaw) and volume spike in uptrend
# Short when Alligator is bearish (Lips < Teeth < Jaw) and volume spike in downtrend
# Volume spike > 1.6x 24-period EMA confirms institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.28 sizing
# Works in bull/bear markets by following the 1d trend direction via EMA50

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h SMAs for Williams Alligator
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().values  # Jaw (13-period)
    teeth = close_s.rolling(window=8, min_periods=8).mean().values   # Teeth (8-period)
    lips = close_s.rolling(window=5, min_periods=5).mean().values    # Lips (5-period)
    
    # Alligator components
    alligator_bullish = (lips > teeth) & (teeth > jaw)   # Bullish alignment
    alligator_bearish = (lips < teeth) & (teeth < jaw)   # Bearish alignment
    
    # Volume confirmation: volume > 1.6 * 24-period EMA
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > (1.6 * vol_ema_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(55, 50, 24, 13)  # Need 1d EMA50, Alligator SMAs, volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # Long: Alligator bullish with volume spike
                if alligator_bullish[i] and volume_spike[i]:
                    signals[i] = 0.28
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # Short: Alligator bearish with volume spike
                if alligator_bearish[i] and volume_spike[i]:
                    signals[i] = -0.28
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid sideways markets
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish (loss of bullish alignment)
            if not alligator_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish (loss of bearish alignment)
            if not alligator_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals