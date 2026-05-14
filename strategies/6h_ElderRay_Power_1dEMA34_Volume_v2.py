#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) with volume > 1.5x 24-bar average and close > 1d EMA34 (uptrend)
# Short when Bear Power < 0 (close < EMA13) with volume > 1.5x 24-bar average and close < 1d EMA34 (downtrend)
# Exit when power crosses zero (Bull Power <= 0 for longs, Bear Power >= 0 for shorts)
# Elder Ray measures bull/bear power relative to EMA13; works in trending markets with pullbacks.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "6h_ElderRay_Power_1dEMA34_Volume_v2"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray Power (13-period EMA) on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13  # Bull Power = Close - EMA13
    bear_power = ema_13 - close  # Bear Power = EMA13 - Close
    
    # Volume confirmation (1.5x 24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(13, 34, 24) + 1  # EMA13 + EMA34(1d) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 with volume spike and close > 1d EMA34 (uptrend)
            if (bull_power[i] > 0 and 
                volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 (Bear Power < 0 in traditional terms) with volume spike and close < 1d EMA34 (downtrend)
            elif (bear_power[i] > 0 and 
                  volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (bullish momentum fading)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 (bearish momentum fading)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals