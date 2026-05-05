#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike
# Long when: Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND volume > 1.5 * avg_volume(20)
# Short when: Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND volume > 1.5 * avg_volume(20)
# Exit when: Alligator turns neutral OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 30-100 total trades over 4 years (7-25/year)
# Alligator identifies trend direction and avoids chop; Elder Ray measures bull/bear power behind the move; volume confirms conviction
# Works in bull markets (strong uptrend with Alligator alignment) and bear markets (strong downtrend with Alligator alignment)

name = "1d_WilliamsAlligator_ElderRay_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with offsets)
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().shift(8).values  # Teeth shifted by 8
    teeth = close_series.rolling(window=8, min_periods=8).mean().shift(5).values   # Lips shifted by 5
    lips = close_series.rolling(window=5, min_periods=5).mean().values
    
    # Calculate Elder Ray (13-period EMA)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish (jaw < teeth < lips) AND Bull Power > 0 AND volume confirmation
            if jaw[i] < teeth[i] and teeth[i] < lips[i] and bull_power[i] > 0 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume confirmation
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and bear_power[i] < 0 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns neutral OR volume drops below average
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns neutral OR volume drops below average
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals