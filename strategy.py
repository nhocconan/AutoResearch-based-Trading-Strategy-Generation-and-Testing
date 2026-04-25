#!/usr/bin/env python3
"""
1d Camarilla H3/L3 Breakout with Weekly EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance.
Breakouts above H3 or below L3 with weekly trend alignment (EMA34) and volume confirmation
capture sustained moves. Works in bull/bear markets by trading breakouts in weekly trend direction.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 1d (based on previous day)
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    prev_close = pd.Series(close).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    
    rang = prev_high - prev_low
    H3 = prev_close + (1.1 * rang / 2)
    L3 = prev_close - (1.1 * rang / 2)
    
    # Volume confirmation: current volume > 1.8 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Weekly EMA34 for trend filter (loaded ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        weekly_trend_up = curr_close > ema_34_1w_aligned[i]
        weekly_trend_down = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND weekly uptrend
            long_entry = (curr_close > H3[i]) and vol_spike and weekly_trend_up
            # Short: price breaks below L3 AND volume spike AND weekly downtrend
            short_entry = (curr_close < L3[i]) and vol_spike and weekly_trend_down
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below L3 (breakdown) OR weekly trend turns down
            if (curr_close < L3[i]) or (not weekly_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (breakout) OR weekly trend turns up
            if (curr_close > H3[i]) or (not weekly_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0