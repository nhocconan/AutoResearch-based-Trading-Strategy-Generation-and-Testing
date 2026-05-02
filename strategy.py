#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d EMA34 trend filter + volume confirmation
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Elder Ray measures bull/bear strength relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong Bull Power (> 0) + price > 1d EMA34 (bullish bias) + volume spike → long entry
# Strong Bear Power (< 0) + price < 1d EMA34 (bearish bias) + volume spike → short entry
# Volume confirmation (2x 20-period average) filters weak breakouts
# Works in bull markets via trend-aligned strength and bear markets via fading weak moves
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_ElderRay_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray (prior completed 6h bar's EMA)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().shift(1).values
    
    # Calculate 6h Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA34 trend (prior completed 1d bar's EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 days for EMA34
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 13, 34)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (bullish strength) AND price > 1d EMA34 (bullish bias) AND volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 (bearish strength) AND price < 1d EMA34 (bearish bias) AND volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (loss of bullish strength) OR price < 1d EMA34 (trend change)
            if bull_power[i] <= 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (loss of bearish strength) OR price > 1d EMA34 (trend change)
            if bear_power[i] >= 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals