#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w EMA34 trend filter
# Uses Williams Alligator (jaw/teeth/lips) for trend direction and Elder Ray (bull/bear power) for momentum confirmation
# 1w EMA34 as higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.5x 24-period average) ensures institutional participation
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Works in bull markets via Alligator alignment + Elder Ray strength, in bear via 1w trend filter avoiding false signals

name = "6h_WilliamsAlligator_ElderRay_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator: SMA(13,8), SMA(8,5), SMA(5,3) - all shifted forward
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line (13,8)
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().shift(5).values    # Red line (8,5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values   # Green line (5,3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13  # Negative values indicate bearish pressure
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and EMA)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # 1w trend filter: only trade in direction of higher timeframe trend
        if close[i] > ema_34_1w_aligned[i]:
            trend_filter = 1   # Bullish higher timeframe
        else:
            trend_filter = -1  # Bearish higher timeframe
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
            alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Bullish Alligator + Bull Power > 0 + volume confirm + 1w bullish trend
            if (alligator_bullish and bull_power[i] > 0 and volume_confirm[i] and 
                trend_filter == 1):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Bear Power < 0 + volume confirm + 1w bearish trend
            elif (alligator_bearish and bear_power[i] < 0 and volume_confirm[i] and 
                  trend_filter == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator loses alignment (Lips < Teeth) or Bull Power turns negative
            if lips[i] < teeth[i] or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator loses alignment (Lips > Teeth) or Bear Power turns positive
            if lips[i] > teeth[i] or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals