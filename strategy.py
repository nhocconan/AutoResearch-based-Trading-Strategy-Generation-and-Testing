#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 4h close > 4h open (bullish 4h candle) AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level AND 4h close < 4h open (bearish 4h candle) AND volume > 1.5x 20-period volume SMA
# - Exit: price reverts to Camarilla Pivot point (PP) or opposite breakout with volume confirmation
# - Uses 4h for signal direction (trend bias) and 1h for precise entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.20 discrete level to control drawdown
# - Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag

name = "1h_4h_camarilla_pivot_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return signals
    
    # Calculate 4h candle direction (bullish/bearish) for trend filter
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    # Bullish 4h candle: close > open
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    # Align to 1h timeframe with proper delay (completed 4h bar only)
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h)
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Camarilla pivot levels (based on previous bar)
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 2
    # L3 = PP - (H - L) * 1.1 / 2
    pp = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3.0
    rng = np.roll(high, 1) - np.roll(low, 1)
    h3 = pp + rng * 1.1 / 2.0
    l3 = pp - rng * 1.1 / 2.0
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(1, n):  # Start from 1 to have previous bar data
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(pp[i]) or
            np.isnan(volume_sma_20[i]) or
            np.isnan(bullish_4h_aligned[i]) or np.isnan(bearish_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > h3[i]  # Break above H3
        breakout_down = close[i] < l3[i]  # Break below L3
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above H3 AND 4h bullish AND volume confirmation
            if breakout_up and bullish_4h_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.20
            # Short: price breaks below L3 AND 4h bearish AND volume confirmation
            elif breakout_down and bearish_4h_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on reversion to Pivot point OR opposite breakout with volume
            exit_condition = (close[i] < pp[i]) or \
                           (breakout_down and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            # Exit on reversion to Pivot point OR opposite breakout with volume
            exit_condition = (close[i] > pp[i]) or \
                           (breakout_up and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals