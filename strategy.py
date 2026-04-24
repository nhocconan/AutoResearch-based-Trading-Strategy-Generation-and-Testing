#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter.
- Donchian(20): 20-period high/low on 1d.
- Entry: Long when price > 20-period high AND price > 1w EMA50 AND volume > 1.5 * 20-period average volume.
         Short when price < 20-period low AND price < 1w EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout (price < 20-period low for long, price > 20-period high for short).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian breakouts capture strong momentum moves.
- 1w EMA50 provides strong long-term trend filter to avoid counter-trend trades.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~60 total over 4 years (~15/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 55  # Need sufficient data for EMA50 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Avoid division by zero
        vol_ratio = curr_volume / (curr_vol_ma + 1e-10)
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price falls below 20-period low
            if position == 1:
                if curr_close < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above 20-period high
            elif position == -1:
                if curr_close > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price > 20-period high AND price > 1w EMA50 AND volume confirmation
            if curr_close > donchian_high[i] and curr_close > ema50_1w_aligned[i] and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price < 20-period low AND price < 1w EMA50 AND volume confirmation
            elif curr_close < donchian_low[i] and curr_close < ema50_1w_aligned[i] and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_TrendFilter_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0