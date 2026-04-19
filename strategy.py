#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Bollinger Band Squeeze Breakout with volume confirmation and weekly trend filter.
# Long when price breaks above upper BB, weekly close > weekly EMA20, and volume > 1.5x 1d average volume.
# Short when price breaks below lower BB, weekly close < weekly EMA20, and volume > 1.5x 1d average volume.
# Exit when price returns to middle BB or volatility expands (BB width > 1.2x 20-day average).
# Uses weekly timeframe for trend filter and Bollinger Bands for volatility-based entry.
# Target: 10-25 trades/year per symbol to stay within frequency limits.
name = "1d_Weekly_BB_Squeeze_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Bollinger Bands (20, 2) on weekly close
    weekly_close = df_1w['close'].values
    sma_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    middle_band = sma_20
    
    # Calculate weekly EMA20 for trend filter
    ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Bollinger Band width for volatility filter
    bb_width = upper_band - lower_band
    avg_bb_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_1w, middle_band)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    avg_bb_width_aligned = align_htf_to_ltf(prices, df_1w, avg_bb_width)
    
    # Get 1d average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(middle_band_aligned[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(avg_bb_width_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = upper_band_aligned[i]
        lower_band = lower_band_aligned[i]
        middle_band = middle_band_aligned[i]
        ema_20 = ema_20_aligned[i]
        bb_width = bb_width_aligned[i]
        avg_bb_width = avg_bb_width_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Volatility filter: only trade when volatility is contracted (squeeze)
        volatility_contracted = bb_width < 0.8 * avg_bb_width
        
        if position == 0:
            # Long entry: price breaks above upper BB, weekly close > weekly EMA20, volume confirmation, volatility squeeze
            if price > upper_band and ema_20 > 0 and volume_confirmed and volatility_contracted:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower BB, weekly close < weekly EMA20, volume confirmation, volatility squeeze
            elif price < lower_band and ema_20 > 0 and volume_confirmed and volatility_contracted:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle BB or volatility expands
            if price < middle_band or bb_width > 1.2 * avg_bb_width:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB or volatility expands
            if price > middle_band or bb_width > 1.2 * avg_bb_width:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals