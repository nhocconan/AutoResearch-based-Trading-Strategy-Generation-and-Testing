#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high + volume > 1.3x 20-period 1d volume SMA + chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low + volume > 1.3x 20-period 1d volume SMA + chop < 61.8 (trending)
# - Exit: price crosses Donchian midpoint (mean reversion)
# - Position sizing: 0.25 discrete level
# - Donchian breakout captures momentum bursts
# - Volume confirmation ensures conviction
# - Chop filter avoids false breakouts in ranging markets
# - Works in bull/bear: breakouts occur in all regimes, chop filter prevents whipsaws

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian channels on 4h timeframe
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index on 1d timeframe (14-period)
    # Chop = 100 * log10(sum(ATR(1)) / (n-period * log(n))) / log10(n)
    # Simplified: Chop = 100 * log10(sum(True Range over period) / (ATR * period)) / log10(period)
    # We'll use standard formula: Chop = 100 * log10(sum(TR) / (ATR * period)) / log10(period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    
    # ATR (14-period)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    chop_1d = np.where(
        (atr_1d > 0) & (sum_tr_14 > 0),
        100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14),
        50  # default to neutral when undefined
    )
    
    # Align HTF indicators to 4h timeframe
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get current 1d volume for confirmation (aligned to 4h)
    vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(vol_1d_current[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period SMA
        vol_confirm = vol_1d_current[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: chop < 61.8 indicates trending market (avoid ranging)
        chop_filter = chop_1d_aligned[i] < 61.8
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Exit conditions: price crosses Donchian midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        if position == 0:  # Flat - look for entry
            if long_breakout and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            elif short_breakout and vol_confirm and chop_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals