#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA34 trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13 of median price, shifted 8 bars), Teeth (EMA8 of median price, shifted 5 bars), Lips (EMA5 of median price, shifted 3 bars)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA34 AND volume > 1.5x 20-bar avg
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA34 AND volume > 1.5x 20-bar avg
# Exit when alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.
# Alligator identifies trending vs ranging markets; 1w EMA34 filters counter-trend moves; volume confirmation ensures participation.
# Works in bull markets (bullish alignment) and bear markets (bearish alignment).

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator components
    median_price = (high + low) / 2.0
    median_series = pd.Series(median_price)
    
    # Jaw: EMA(13) of median price, shifted 8 bars
    jaw = median_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift right by 8 (look back)
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: EMA(8) of median price, shifted 5 bars
    teeth = median_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5 (look back)
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: EMA(5) of median price, shifted 3 bars
    lips = median_series.ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift right by 3 (look back)
    lips[:3] = np.nan  # first 3 values invalid
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34)  # Alligator jaw warmup and EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1w_aligned[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        prev_lips = lips[i-1]
        prev_teeth = teeth[i-1]
        prev_jaw = jaw[i-1]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: bullish alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bearish alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA34 AND volume confirmation
            if curr_lips > curr_teeth and curr_teeth > curr_jaw and close[i] > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA34 AND volume confirmation
            elif curr_lips < curr_teeth and curr_teeth < curr_jaw and close[i] < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals