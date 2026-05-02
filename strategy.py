#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets: buy when Bull Power > 0 and rising, price > 1d EMA34 (uptrend)
# In bear markets: sell when Bear Power < 0 and falling, price < 1d EMA34 (downtrend)
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Works in both bull and bear markets by taking trend-aligned Elder Ray signals
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe

name = "6h_ElderRay_1dEMA34_Trend_Volume_v1"
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
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) for trend determination
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND rising (current > previous) + volume confirm + price > 1d EMA34 (bullish trend)
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND falling (current < previous) + volume confirm + price < 1d EMA34 (bearish trend)
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (momentum lost) OR price < 1d EMA34 (trend reversal)
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (momentum lost) OR price > 1d EMA34 (trend reversal)
            if bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals