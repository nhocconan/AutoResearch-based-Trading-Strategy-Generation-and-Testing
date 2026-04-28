#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trending vs ranging markets.
# In strong trends (Alligator aligned), we trade breakouts in direction of trend.
# 1d EMA34 provides higher timeframe trend bias to avoid counter-trend trades.
# Volume confirmation ensures breakouts have momentum.
# Position size 0.25 for balanced risk. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (6h timeframe)
    # JAW: 13-period SMMA, shifted 8 bars ahead
    # TEETH: 8-period SMMA, shifted 5 bars ahead  
    # LIPS: 5-period SMMA, shifted 3 bars ahead
    # Using EMA as proxy for SMMA (similar smoothing properties)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment check
        # Bullish alignment: Lips > Teeth > Jaw (green alignment)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Jaw > Teeth > Lips (red alignment)
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions: price outside Alligator mouth
        # Long: price above highest of Alligator components
        # Short: price below lowest of Alligator components
        alligator_high = np.maximum(jaw[i], np.maximum(teeth[i], lips[i]))
        alligator_low = np.minimum(jaw[i], np.minimum(teeth[i], lips[i]))
        price_above_alligator = close[i] > alligator_high
        price_below_alligator = close[i] < alligator_low
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry conditions: Alligator aligned + price breakout + volume + HTF trend
        long_entry = bullish_alignment and price_above_alligator and price_above_ema and vol_confirm
        short_entry = bearish_alignment and price_below_alligator and price_below_ema and vol_confirm
        
        # Exit conditions: price re-enters Alligator mouth or opposite signal
        long_exit = close[i] < alligator_high
        short_exit = close[i] > alligator_low
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals