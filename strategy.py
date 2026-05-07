#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator + 1-day trend filter with volume confirmation.
# Long when: Jaw < Teeth < Lips (bullish alignment) AND EMA34(1d) rising AND volume > 1.5 * EMA50(volume).
# Short when: Jaw > Teeth > Lips (bearish alignment) AND EMA34(1d) falling AND volume > 1.5 * EMA50(volume).
# Exit when Alligator lines re-cross (Jaw crosses Teeth).
# Williams Alligator identifies trend alignment; 1-day EMA34 filters higher timeframe trend;
# Volume confirmation ensures breakout strength. Designed for low trade frequency (target: 15-30/year)
# to minimize fee drag and improve generalization across bull/bear markets.
name = "6h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator lines (13, 8, 5 SMAs shifted forward by 8, 5, 3 bars)
    # Jaw (13-period SMA, shifted 8 bars)
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth (8-period SMA, shifted 5 bars)
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips (5-period SMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Bullish alignment: Jaw < Teeth < Lips
    bullish_align = (jaw_vals < teeth_vals) & (teeth_vals < lips_vals)
    # Bearish alignment: Jaw > Teeth > Lips
    bearish_align = (jaw_vals > teeth_vals) & (teeth_vals > lips_vals)
    
    # EMA34 on 1d close for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.5 * 50-period EMA of volume
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_conf = volume > (1.5 * vol_ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish alignment AND EMA34(1d) rising AND volume confirmation
            long_condition = bullish_align[i] and ema_34_rising_aligned[i] and volume_conf[i]
            # Short: Bearish alignment AND EMA34(1d) falling AND volume confirmation
            short_condition = bearish_align[i] and ema_34_falling_aligned[i] and volume_conf[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bearish alignment (Jaw > Teeth)
            if jaw_vals[i] > teeth_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bullish alignment (Jaw < Teeth)
            if jaw_vals[i] < teeth_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals