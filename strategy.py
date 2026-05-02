#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray Power with 1d EMA34 trend filter and volume confirmation
# Williams Alligator (jaw/teeth/lips) identifies trend absence when intertwined; Elder Ray Power measures
# bull/bear strength via EMA13. Combined with 1d EMA34 for higher-timeframe trend alignment and volume
# spike confirmation, this strategy aims to capture strong directional moves in both bull and bear markets
# while avoiding choppy regimes. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsAlligator_ElderRay_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - SMMA of median price
    # Using close as proxy for median price; jaw=13-period, teeth=8-period, lips=5-period
    close_s = pd.Series(close)
    alligator_jaw = close_s.rolling(window=13, min_periods=13).mean().values
    alligator_teeth = close_s.rolling(window=8, min_periods=8).mean().values
    alligator_lips = close_s.rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator jaws and EMA13)
    start_idx = 13
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(alligator_jaw[i]) or np.isnan(alligator_teeth[i]) or np.isnan(alligator_lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator condition: trend present when lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        alligator_bullish = alligator_lips[i] > alligator_teeth[i] > alligator_jaw[i]
        alligator_bearish = alligator_lips[i] < alligator_teeth[i] < alligator_jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish AND Elder Ray bull power positive AND price > 1d EMA34 AND volume spike
            if (alligator_bullish and 
                bull_power[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish AND Elder Ray bear power negative AND price < 1d EMA34 AND volume spike
            elif (alligator_bearish and 
                  bear_power[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Elder Ray bull power turns negative OR price < 1d EMA34
            if (not alligator_bullish or bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Elder Ray bear power turns positive OR price > 1d EMA34
            if (not alligator_bearish or bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals