#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter, volume spike confirmation, and choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend filter to capture intermediate trend direction.
- Donchian(20): Price channel breakout identifies momentum bursts.
- Entry: Long when price breaks above Donchian(20) upper band AND price > 12h EMA34 AND volume > 2.0 * 20-period average volume AND choppiness < 61.8 (trending regime).
         Short when price breaks below Donchian(20) lower band AND price < 12h EMA34 AND volume > 2.0 * 20-period average volume AND choppiness < 61.8.
- Exit: Opposite Donchian breakout OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Choppiness filter (CHOP < 61.8) ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- 12h EMA34 provides trend alignment to avoid counter-trend trades.
- Estimated trades: ~120 total over 4 years (~30/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period=20):
    """Calculate Donchian Channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr = pd.Series(0.0, index=range(len(high)))
    for i in range(len(high)):
        atr.iloc[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]) if i > 0 else 0,
            abs(low[i] - close[i-1]) if i > 0 else 0
        )
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h, additional_delay_bars=1)
    
    # Donchian(20) channels
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    # Choppiness Index (14-period) for regime filter
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_donch_upper = donch_upper[i]
        curr_donch_lower = donch_lower[i]
        curr_chop = chop[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian lower band OR price falls below 12h EMA34
            if position == 1:
                if curr_close < curr_donch_lower or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper band OR price rises above 12h EMA34
            elif position == -1:
                if curr_close > curr_donch_upper or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter, volume confirmation, and chop filter
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume (using 12h average)
            vol_ma_20_current = vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else 0
            volume_confirmed = curr_volume > 2.0 * vol_ma_20_current
            
            # Choppiness filter: only trade in trending markets (CHOP < 61.8)
            chop_filter = curr_chop < 61.8
            
            # Long: Donchian upper breakout AND price > 12h EMA34 AND volume confirmation AND chop filter
            if curr_close > curr_donch_upper and curr_close > ema34_12h_aligned[i] and volume_confirmed and chop_filter:
                signals[i] = 0.30
                position = 1
            # Short: Donchian lower breakout AND price < 12h EMA34 AND volume confirmation AND chop filter
            elif curr_close < curr_donch_lower and curr_close < ema34_12h_aligned[i] and volume_confirmed and chop_filter:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA34_TrendFilter_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0