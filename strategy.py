#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Williams Alligator: Jaw (EMA13 of median price, 8-bar shift), Teeth (EMA8 of median price, 5-bar shift), Lips (EMA5 of median price, 3-bar shift)
# Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period MA
# Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Alligator alignment or EMA50 trend reversal.
# Discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Alligator identifies trend via smoothed median price lines; 1w EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment. Low trade frequency due to strict alignment conditions.

name = "1d_WilliamsAlligator_1wEMA50_Volume"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: using median price (hl2)
    hl2 = (high + low) / 2.0
    
    # Jaw: EMA13 of hl2, 8-bar shift
    jaw = pd.Series(hl2).ewm(span=13, min_periods=13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)  # shift right by 8 bars
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: EMA8 of hl2, 5-bar shift
    teeth = pd.Series(hl2).ewm(span=8, min_periods=8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5 bars
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: EMA5 of hl2, 3-bar shift
    lips = pd.Series(hl2).ewm(span=5, min_periods=5, adjust=False).mean().values
    lips = np.roll(lips, 3)  # shift right by 3 bars
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Volume regime: current 1d volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        l = lips[i]
        t = teeth[i]
        j = jaw[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Alligator alignment
        bullish_alignment = l > t and t > j  # Lips > Teeth > Jaw
        bearish_alignment = l < t and t < j  # Lips < Teeth < Jaw
        
        # Entry logic
        if position == 0:
            # Long: Bullish alignment AND uptrend AND volume spike
            if bullish_alignment and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND downtrend AND volume spike
            elif bearish_alignment and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alignment (loss of bullish momentum) OR trend turns down
            if bearish_alignment or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment (loss of bearish momentum) OR trend turns up
            if bullish_alignment or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals