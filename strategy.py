#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volume spike filter and trend confirmation.
- Primary timeframe: 4h targeting 100-200 total trades over 4 years (25-50/year).
- HTF: 1d for ATR-based volume spike filter and EMA50 trend filter.
- Donchian Channel: Upper/lower bands from 20-period high/low on 4h for breakout.
- Volume Spike Filter: Current 4h volume > 1.5 * 20-period average 4h volume AND 
                       1d ATR(14) > 1.2 * 20-period average 1d ATR(14) (volatility expansion).
- Trend Filter: 4h close > EMA50 for longs, close < EMA50 for shorts.
- Entry: Long when price breaks above Donchian upper AND volume spike AND uptrend.
         Short when price breaks below Donchian lower AND volume spike AND downtrend.
- Exit: Opposite Donchian break (long exits when price < lower band, short exits when price > upper band).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture volatility expansion breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 4h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 4h volume MA for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient 1d data
        return np.zeros(n)
    
    # 1d True Range calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR MA for volatility comparison
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and ATR MA to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 50, 20)  # Donchian20, EMA50, volMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema50[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_up = donchian_upper[i]
        donchian_low = donchian_lower[i]
        ema50_val = ema50[i]
        vol_ma_val = vol_ma[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_ma_1d_val = atr_ma_1d_aligned[i]
        
        # Donchian breakout conditions
        broke_above = curr_high > donchian_up  # Use high for breakout detection
        broke_below = curr_low < donchian_low   # Use low for breakout detection
        
        # Volume spike filter: 4h volume > 1.5 * 20-period average AND 1d ATR > 1.2 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma_val
        atr_expansion = atr_1d_val > 1.2 * atr_ma_1d_val
        volatility_filter = volume_spike and atr_expansion
        
        # Trend filter: EMA50 direction
        uptrend = curr_close > ema50_val
        downtrend = curr_close < ema50_val
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: price breaks below Donchian lower
            if position == 1:
                if curr_low < donchian_low:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper
            elif position == -1:
                if curr_high > donchian_up:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volatility and trend filters
        if position == 0:
            # Long: break above upper AND volatility filter AND uptrend
            long_condition = broke_above and volatility_filter and uptrend
            
            # Short: break below lower AND volatility filter AND downtrend
            short_condition = broke_below and volatility_filter and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRVolSpike_EMA50Trend_v1"
timeframe = "4h"
leverage = 1.0