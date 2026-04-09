#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams Alligator (Jaw/Teeth/Lips) with 1w HTF trend filter and volume confirmation
# - Uses 1w HTF for trend direction: price > EMA8(1w) = uptrend, price < EMA8(1w) = downtrend
# - Williams Alligator on 1d: Jaw=EMA(13,8), Teeth=EMA(8,5), Lips=EMA(5,3)
# - Long when Lips > Teeth > Jaw (bullish alignment) + price > Teeth
# - Short when Lips < Teeth < Jaw (bearish alignment) + price < Teeth
# - Volume confirmation: current 1d volume > 1.3x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 20-50 trades/year on 1d timeframe (80-200 total over 4 years)

name = "1d_1w_williams_alligator_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA8 for trend filter
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Williams Alligator on 1d
    # Jaw: EMA(13, 8 periods)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Teeth: EMA(8, 5 periods)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    # Lips: EMA(5, 3 periods)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_8_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema_8_1w_aligned[i]
        downtrend_1w = close[i] < ema_8_1w_aligned[i]
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when bullish alignment breaks or price < Teeth
            if not bullish_alignment or close[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when bearish alignment breaks or price > Teeth
            if not bearish_alignment or close[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic
            if volume_confirmed:
                if bullish_alignment and uptrend_1w and close[i] > teeth[i]:
                    position = 1
                    signals[i] = position_size
                elif bearish_alignment and downtrend_1w and close[i] < teeth[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals