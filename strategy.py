#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) + 1w EMA34 Trend Filter + Volume Confirmation
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA34 AND volume > 1.5x avg
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA34 AND volume > 1.5x avg
# Uses discrete sizing (0.25) to minimize fee churn. Alligator identifies trend, 1w EMA filter ensures higher-timeframe alignment.
# Timeframe: 1d (primary), HTF: 1w for EMA34 trend.

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
    
    # Load HTF data ONCE before loop for 1w EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator components (SMA with specific offsets)
    # Jaw: SMA(13, 8) -> 13-period SMA shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8, 5) -> 8-period SMA shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5, 3) -> 5-period SMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13+8, 8+5, 5+3, 20, 34)  # warmup for Alligator, volume MA, HTF EMA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            # 2. Price falls below 1w EMA34 (trend change)
            if (curr_lips <= curr_teeth or
                curr_teeth <= curr_jaw or
                curr_close < curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            # 2. Price rises above 1w EMA34 (trend change)
            if (curr_lips >= curr_teeth or
                curr_teeth >= curr_jaw or
                curr_close > curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish alignment (Lips > Teeth > Jaw) AND price > 1w EMA34 AND volume confirm
            if (curr_lips > curr_teeth and
                curr_teeth > curr_jaw and
                curr_close > curr_ema_34_1w and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment (Lips < Teeth < Jaw) AND price < 1w EMA34 AND volume confirm
            elif (curr_lips < curr_teeth and
                  curr_teeth < curr_jaw and
                  curr_close < curr_ema_34_1w and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals