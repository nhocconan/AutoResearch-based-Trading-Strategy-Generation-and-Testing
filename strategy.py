#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator (13,8,5 SMAs) on 6h defines market state: 
#   * Alligator asleep (JAW > TEETH > LIPS or JAW < TEETH < LIPS) = ranging market
#   * Alligator awake (JAW < TEETH < LIPS) = uptrend
#   * Alligator awake (JAW > TEETH > LIPS) = downtrend
# - 1d EMA(50) as trend filter: only take Alligator signals in direction of higher timeframe trend
# - 1d volume > 1.5x 20-period average for conviction
# - Entry: Alligator awake + 1d trend alignment + volume confirmation
# - Exit: Alligator returns to sleep (JAW crosses TEETH or LIPS) or trend reversal
# - Position size: 0.25 to manage drawdown in volatile 6h markets
# - Designed to catch trends while avoiding whipsaws in ranging markets
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "6h_WilliamsAlligator_1dTrend_Volume_v1"
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
    
    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    
    # Williams Alligator: Jaw (13-period SMA), Teeth (8-period SMA), Lips (5-period SMA)
    # All SMAs are calculated on median price (high+low)/2 with future shift
    median_price = (df_6h['high'] + df_6h['low']) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 6s timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session filter (00:00-23:00 UTC - trade all hours for 6h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    # Always true for 6h - trade all sessions
    in_session = np.ones(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Williams Alligator conditions
        # Alligator asleep: JAW > TEETH > LIPS (downtrend) or JAW < TEETH < LIPS (uptrend but weak)
        # Actually, sleeping is when intertwined: not (JAW > TEETH > LIPS) and not (JAW < TEETH < LIPS)
        # Awake is when JAW, TEETH, LIPS are properly aligned
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Alligator awake and aligned: JAW < TEETH < LIPS = uptrend
        alligator_awake_up = jaw_val < teeth_val < lips_val
        # Alligator awake and aligned: JAW > TEETH > LIPS = downtrend
        alligator_awake_down = jaw_val > teeth_val > lips_val
        # Alligator sleeping/intertwined: otherwise
        alligator_asleep = not (alligator_awake_up or alligator_awake_down)
        
        # Volume filter: current volume > 1.5x 1d average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # 1d trend filter
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: alligator awake up + 1d uptrend + volume
            if alligator_awake_up and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: alligator awake down + 1d downtrend + volume
            elif alligator_awake_down and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on alligator sleeping or 1d trend reversal
            if alligator_asleep or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on alligator sleeping or 1d trend reversal
            if alligator_asleep or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals