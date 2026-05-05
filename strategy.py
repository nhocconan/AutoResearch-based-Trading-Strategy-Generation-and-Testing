#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Alligator Jaw = EMA(close, 13) shifted 8 bars
# Alligator Teeth = EMA(close, 8) shifted 5 bars
# Alligator Lips = EMA(close, 5) shifted 3 bars
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA34 AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA34 AND volume > 1.5x 20-period average
# Exit when alignment breaks: not (Lips > Teeth > Jaw) for long OR not (Lips < Teeth < Jaw) for short
# Williams Alligator identifies trend absence/presence; effective in ranging (avoid false signals) and trending (ride trends) markets.
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_WilliamsAlligator_1wEMA34_Volume"
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
    
    # Calculate volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator components on 1d
    # Jaw = EMA(close, 13) shifted 8 bars
    # Teeth = EMA(close, 8) shifted 5 bars
    # Lips = EMA(close, 5) shifted 3 bars
    if len(close) >= 13:
        ema_5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
        ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        
        lips = np.roll(ema_5, 3)
        teeth = np.roll(ema_8, 5)
        jaw = np.roll(ema_13, 8)
        
        # Handle NaN from rolling
        lips[:3] = np.nan
        teeth[:5] = np.nan
        jaw[:8] = np.nan
    else:
        lips = np.full(n, np.nan)
        teeth = np.full(n, np.nan)
        jaw = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND volume filter AND above 1w EMA34
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND volume filter AND below 1w EMA34
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (not Lips > Teeth > Jaw)
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (not Lips < Teeth < Jaw)
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals