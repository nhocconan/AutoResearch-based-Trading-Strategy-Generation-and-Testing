#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d trend filter, volume spike, and choppiness regime.
- Primary timeframe: 4h for lower trade frequency and reduced fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to confirm institutional interest.
- Donchian: 20-period upper/lower bands for breakout signals.
- Choppiness: CHOP(14) > 61.8 for ranging market (mean reversion), < 38.2 for trending.
- Entry: Long when price breaks above Donchian(20) upper band AND 1d EMA34 bullish AND volume spike AND CHOP < 38.2 (trending).
         Short when price breaks below Donchian(20) lower band AND 1d EMA34 bearish AND volume spike AND CHOP < 38.2 (trending).
- Exit: Opposite Donchian band touch or loss of volume confirmation or regime shift to chop (CHOP > 61.8).
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
This strategy captures strong trending moves with volume confirmation while avoiding choppy markets,
providing robustness in both bull and bear regimes by only trading in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) channels
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 4h volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate 4h Choppiness Index(14)
    chop_window = 14
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).rolling(window=chop_window, min_periods=1).mean().values
    atr[0] = high[0] - low[0]  # fix first value
    sum_atr = pd.Series(atr).rolling(window=chop_window, min_periods=chop_window).sum().values
    highest_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    lowest_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    chop = 100 * np.log10(sum_atr / np.log10(chop_window)) / np.log10((highest_high - lowest_low) + 1e-10)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20, 34, chop_window)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        ema_val = ema_1d_aligned[i]
        chop_val = chop[i]
        
        if position == 0:
            # Check for entry signals with volume spike and trending regime (CHOP < 38.2)
            if volume_spike[i] and chop_val < 38.2:
                # Bullish: break above Donchian upper band AND 1d EMA34 bullish (close > EMA)
                if curr_high > dch_high and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: break below Donchian lower band AND 1d EMA34 bearish (close < EMA)
                elif curr_low < dch_low and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: touch Donchian lower band OR loss of volume confirmation OR regime shift to chop (CHOP > 61.8)
            if curr_low <= dch_low or not volume_spike[i] or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: touch Donchian upper band OR loss of volume confirmation OR regime shift to chop (CHOP > 61.8)
            if curr_high >= dch_high or not volume_spike[i] or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0