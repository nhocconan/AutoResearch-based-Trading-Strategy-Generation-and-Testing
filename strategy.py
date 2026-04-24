#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume average and Camarilla pivot calculation, 1w for EMA trend filter.
- Camarilla Pivots: identifies key support/resistance levels from prior day.
- Entry: Long when price breaks above H3 level AND volume > 1.5 * 20-period average volume AND price > 1w EMA34.
         Short when price breaks below L3 level AND volume > 1.5 * 20-period average volume AND price < 1w EMA34.
- Exit: Opposite Camarilla breakout signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla breakouts capture institutional order flow reactions.
- Volume confirmation ensures breakout legitimacy.
- Weekly EMA filter ensures trading with higher timeframe trend.
- Works in both bull and bear markets as it captures volatility expansion after contraction at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for the period."""
    typical_price = (high + low + close) / 3.0
    range_ = high - low
    H4 = close + range_ * 1.1 / 2
    H3 = close + range_ * 1.1 / 4
    H2 = close + range_ * 1.1 / 6
    H1 = close + range_ * 1.1 / 12
    L1 = close - range_ * 1.1 / 12
    L2 = close - range_ * 1.1 / 6
    L3 = close - range_ * 1.1 / 4
    L4 = close - range_ * 1.1 / 2
    return H3, L3  # We only need H3 and L3 for breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla H3/L3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 1d bar
    H3_1d = np.zeros(len(df_1d))
    L3_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        H3_1d[i], L3_1d[i] = camarilla_pivot(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i]
        )
    
    # Align H3/L3 to 12h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # Calculate 1d volume average for confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below L3 level
            if position == 1:
                if curr_low <= L3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 level
            elif position == -1:
                if curr_high >= H3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and weekly trend filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= H3_1d_aligned[i] and prev_close < H3_1d_aligned[i-1]
            breakout_down = curr_low <= L3_1d_aligned[i] and prev_close > L3_1d_aligned[i-1]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Weekly trend filter: price above/below weekly EMA34
            uptrend = curr_close > ema_34_1w_aligned[i]
            downtrend = curr_close < ema_34_1w_aligned[i]
            
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dVolumeSpike_1wEMA34_v1"
timeframe = "12h"
leverage = 1.0