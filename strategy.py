#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 Trend Filter + Volume Confirmation
# Williams Alligator: Jaw (EMA13, 8-period smoothed), Teeth (EMA8, 5-period smoothed), Lips (EMA5, 3-period smoothed)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 1.8x avg
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 1.8x avg
# Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter.
# Timeframe: 12h (primary), HTF: 1d for EMA34 trend.

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (all using close prices)
    # Jaw: EMA13 smoothed by 8 periods
    ema_jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(ema_jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA8 smoothed by 5 periods
    ema_teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(ema_teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA5 smoothed by 3 periods
    ema_lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(ema_lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 8, 5, 20)  # warmup for EMA34, Jaw, Teeth, Lips, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            # 2. Price falls below 1d EMA34 (trend change)
            if (curr_lips <= curr_teeth or
                curr_teeth <= curr_jaw or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            # 2. Price rises above 1d EMA34 (trend change)
            if (curr_lips >= curr_teeth or
                curr_teeth >= curr_jaw or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume confirm
            if (curr_lips > curr_teeth and
                curr_teeth > curr_jaw and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume confirm
            elif (curr_lips < curr_teeth and
                  curr_teeth < curr_jaw and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals