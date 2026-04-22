#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + Elder Ray Power with 1d EMA50 trend
    # Works in both bull and bear markets: Alligator identifies trend direction,
    # Elder Ray measures bull/bear power, EMA50 filters for higher timeframe trend
    # Low trade frequency by design: Alligator requires clear separation of SMAs
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 trend filter
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Williams Alligator: three SMAs (Jaw=13, Teeth=8, Lips=5)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND price above daily EMA50
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and bull_power[i] > 0 and close[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) AND Bear Power < 0 AND price below daily EMA50
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and bear_power[i] < 0 and close[i] < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines re-cross or power reverses
            if position == 1:
                if lips[i] < teeth[i] or bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i] or bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_ElderRay_Power_1dEMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0