#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 AND 1d EMA34 uptrend AND volume > 2.0 * volume_ma(30)
- Short when price breaks below Camarilla S1 AND 1d EMA34 downtrend AND volume > 2.0 * volume_ma(30)
- Uses Camarilla pivot levels from prior completed 1d bar for structure-based breakouts
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (2.0x) confirms strong institutional participation and reduces false breakouts
- Designed for low frequency (target 15-40 trades/year) to minimize fee drag and improve test generalization
- Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal
- Novelty: Tight volume confirmation (2.0x) and 1d trend filter specifically for BTC/ETH edge in both bull/bear markets
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
    
    # Load 4h data ONCE before loop for Camarilla pivot calculation (structure)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels from prior completed 1d bar
    # Need 1d OHLC from 4h data - resample 4h to get daily OHLC
    # Since we cannot resample, we'll use the last 4h bar of each day as proxy
    # Better approach: load 1d data directly for pivot calculation
    
    # Load 1d data ONCE before loop for accurate Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels using prior 1d bar (HLC of completed daily candle)
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    hlc_1d = df_1d[['high', 'low', 'close']]
    prev_high = hlc_1d['high'].shift(1).values  # Prior completed 1d high
    prev_low = hlc_1d['low'].shift(1).values    # Prior completed 1d low
    prev_close = hlc_1d['close'].shift(1).values # Prior completed 1d close
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to complete)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    # Reuse df_1d from above
    
    # Calculate 1d EMA34 for trend filter (needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(30) for confirmation
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 30 for volume MA)
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 1d uptrend AND volume spike
            if close[i] > camarilla_r1_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND 1d downtrend AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR 1d trend turns down
            if close[i] < camarilla_s1_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR 1d trend turns up
            if close[i] > camarilla_r1_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0