#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power (close - EMA13) to detect strength
# Williams Alligator (Jaw/Teeth/Lips) identifies trend vs ranging markets
# In trending markets (Alligator aligned): trade Elder Ray direction
# In ranging markets (Alligator intertwined): fade extreme Elder Ray readings
# Volume confirmation (1.5x 20-period average) filters low-participation moves
# Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_ElderRay_Alligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Elder Ray components (13-period EMA)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13  # Bull Power = Close - EMA13
    bear_power = ema_13 - close  # Bear Power = EMA13 - Close
    
    # Calculate Williams Alligator (13,8,5 SMAs with offsets)
    # Jaw: 13-period SMA, offset 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, offset 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, offset 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine market state using Alligator
        # Trending: Jaw > Teeth > Lips (uptrend) or Jaw < Teeth < Lips (downtrend)
        # Ranging: Alligator lines intertwined (not clearly separated)
        is_uptrend = jaw[i] > teeth[i] and teeth[i] > lips[i]
        is_downtrend = jaw[i] < teeth[i] and teeth[i] < lips[i]
        is_ranging = not (is_uptrend or is_downtrend)
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: go long on strong bull power
                if bull_power[i] > 0 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend:
                # In downtrend: go short on strong bear power
                if bear_power[i] > 0 and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # ranging market
                # In range: fade extreme Elder Ray readings
                if bull_power[i] > np.percentile(bull_power[max(0, i-50):i+1], 80) and volume_confirm[i]:
                    # Extreme bull power in range -> short (mean reversion)
                    signals[i] = -0.25
                    position = -1
                elif bear_power[i] > np.percentile(bear_power[max(0, i-50):i+1], 80) and volume_confirm[i]:
                    # Extreme bear power in range -> long (mean reversion)
                    signals[i] = 0.25
                    position = 1
            # Default: stay flat
            if position == 0:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if is_uptrend:
                # In uptrend: exit when bull power fades
                if bull_power[i] <= 0:
                    exit_signal = True
            else:
                # In ranging/downtrend: exit on any reversal signal
                if bear_power[i] > bull_power[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if is_downtrend:
                # In downtrend: exit when bear power fades
                if bear_power[i] <= 0:
                    exit_signal = True
            else:
                # In ranging/uptrend: exit on any reversal signal
                if bull_power[i] > bear_power[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals