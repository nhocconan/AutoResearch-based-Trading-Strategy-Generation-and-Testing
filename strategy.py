#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA34 AND volume > 1.5x 24-bar avg
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA34 AND volume > 1.5x 24-bar avg
# Exit when Alligator alignment reverses or volume drops below average
# Williams Alligator identifies trending vs ranging markets via smoothed medians (13,8,5 periods).
# 1d EMA34 ensures we trade with the dominant daily trend, improving win rate in both bull/bear markets.
# Volume confirmation filters weak breakouts. Discrete sizing (0.25) minimizes fee drag.
# Target: 12-25 trades/year on 12h (50-100 total over 4 years).

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    # We'll use EMA as proxy for SMMA as it's commonly accepted
    
    median_12h = (high + low) / 2.0  # Typical price for Alligator
    
    jaw = pd.Series(median_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply shifts (Alligator shifts jaws/teeth/lips into future)
    # Jaw shifted 8 bars, teeth 5 bars, lips 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill NaN from roll with first valid values
    jaw_shifted[:8] = jaw[7] if not np.isnan(jaw[7]) else 0
    teeth_shifted[:5] = teeth[4] if not np.isnan(teeth[4]) else 0
    lips_shifted[:3] = lips[2] if not np.isnan(lips[2]) else 0
    
    # Volume confirmation: >1.5x 24-bar average volume (2 days on 12h)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Volume MA needs 24 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Bullish alignment: jaw < teeth < lips
        bullish = jaw_val < teeth_val < lips_val
        # Bearish alignment: jaw > teeth > lips
        bearish = jaw_val > teeth_val > lips_val
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when bullish alignment AND price > 1d EMA34 AND volume confirmation
            if bullish and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND price < 1d EMA34 AND volume confirmation
            elif bearish and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when alignment turns bearish or volume drops
            if not bullish or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when alignment turns bullish or volume drops
            if not bearish or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals