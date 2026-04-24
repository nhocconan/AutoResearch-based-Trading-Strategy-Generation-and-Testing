#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for trend filter (price > SMA50 + 0.5*ATR = uptrend, price < SMA50 - 0.5*ATR = downtrend).
- Entry: Long when price breaks above Donchian(20) high AND price > SMA50 + 0.5*ATR AND volume > 2.0 * 12h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < SMA50 - 0.5*ATR AND volume > 2.0 * 12h volume MA(20).
- Exit: Opposite Donchian breakout (Long exits when price < Donchian(20) low, Short exits when price > Donchian(20) high).
- Signal size: 0.25 discrete to balance capture and fee control.
- Uses Donchian structure for clear breakouts, ATR-based trend filter adapts to volatility, volume spike confirms conviction.
- Works in bull (buying strong breakouts) and bear (selling strong breakdowns) with volatility-adjusted trend filter.
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
    
    # Get 1d data for ATR(14) and SMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d SMA50
    close_1d = df_1d['close'].values
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate trend bands: SMA50 ± 0.5*ATR
    upper_band = sma_50 + 0.5 * atr_14
    lower_band = sma_50 - 0.5 * atr_14
    
    # Align trend bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Get 12h data for Donchian(20) channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Get 12h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > upper_band = uptrend, price < lower_band = downtrend
        uptrend = curr_close > upper_band_aligned[i]
        downtrend = curr_close < lower_band_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Donchian high
                if curr_high > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Donchian low
                if curr_low < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Donchian low
            if curr_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian high
            if curr_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0