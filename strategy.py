#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter (EMA34) and volume confirmation.
# Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift).
# Long when Lips > Teeth > Jaw (bullish alignment) and price > Lips, with volume > 1.5x 20-period MA.
# Short when Lips < Teeth < Jaw (bearish alignment) and price < Lips, with volume spike.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "12h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h
    # Jaw: EMA13, 8-bar shift
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: EMA8, 5-bar shift
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: EMA5, 3-bar shift
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Entry logic
        if position == 0:
            # Long: bullish alignment AND price > Lips AND 1d uptrend AND volume spike
            if bullish_alignment and close_val > lips[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < Lips AND 1d downtrend AND volume spike
            elif bearish_alignment and close_val < lips[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR price < Lips OR 1d trend turns down
            if bearish_alignment or close_val < lips[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR price > Lips OR 1d trend turns up
            if bullish_alignment or close_val > lips[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals