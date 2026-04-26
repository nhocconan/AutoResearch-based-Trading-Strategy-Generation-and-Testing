#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 level AND 4h EMA34 uptrend AND volume > 1.8 * volume_ma(20)
- Short when price breaks below Camarilla S1 level AND 4h EMA34 downtrend AND volume > 1.8 * volume_ma(20)
- Uses Camarilla pivot levels from 1h chart for structure-based breakouts
- 4h EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (1.8x) confirms participation and reduces false breakouts
- Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal
- Designed for moderate frequency (target 15-37 trades/year on 1h) to minimize fee drag
- Novelty: Applying proven Camarilla R1/S1 breakout logic to 1h timeframe with 4h trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter (needs completed 4h candle)
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_4h = np.where(ema_34_4h_aligned > 0, 
                        np.where(close > ema_34_4h_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels on 1h chart (primary timeframe)
    # Using previous bar's OHLC for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels
    R1 = pivot + (range_hl * 1.1 / 12.0)
    R2 = pivot + (range_hl * 1.1 / 6.0)
    # Support levels
    S1 = pivot - (range_hl * 1.1 / 12.0)
    S2 = pivot - (range_hl * 1.1 / 6.0)
    
    # Calculate volume filter: volume > 1.8 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 4h EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(trend_4h[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla R1/S1 breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 4h uptrend AND volume spike
            if close[i] > R1[i] and trend_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S1 AND 4h downtrend AND volume spike
            elif close[i] < S1[i] and trend_4h[i] == -1 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below Camarilla S1 OR 4h trend turns down
            if close[i] < S1[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above Camarilla R1 OR 4h trend turns up
            if close[i] > R1[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0