#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 level AND 12h EMA50 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Camarilla S1 level AND 12h EMA50 downtrend AND volume > 1.5 * volume_ma(20)
- Uses Camarilla pivot levels from completed 4h bars for structure-based breakouts
- 12h EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 19-50 trades/year) to minimize fee drag
- Exit on opposite Camarilla level touch or trend reversal
- Novelty: Combines Camarilla breakouts with HTF trend and volume confirmation for BTC/ETH edge in both bull/bear markets
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
    
    # Load 4h data ONCE before loop for Camarilla levels (structure)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels from prior 4h bar (completed bar only)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    hl_range = df_4h['high'] - df_4h['low']
    camarilla_r1 = typical_price + (hl_range * 1.1 / 12)
    camarilla_s1 = typical_price - (hl_range * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed for structure)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1.values)
    
    # Load 12h data ONCE before loop for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter (needs completed 12h candle)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_12h = np.where(ema_50_12h_aligned > 0, 
                         np.where(close > ema_50_12h_aligned, 1, -1), 
                         0)
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(trend_12h[i]) or np.isnan(volume_ma[i])):
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
            # Long: Price breaks above Camarilla R1 AND 12h uptrend AND volume spike
            if close[i] > camarilla_r1_aligned[i] and trend_12h[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND 12h downtrend AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and trend_12h[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR 12h trend turns down
            if close[i] < camarilla_s1_aligned[i] or trend_12h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR 12h trend turns up
            if close[i] > camarilla_r1_aligned[i] or trend_12h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0