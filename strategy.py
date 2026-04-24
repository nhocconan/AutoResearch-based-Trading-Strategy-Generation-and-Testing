#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4L4 Breakout + 12h HMA21 Trend Filter + Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h HMA21 for trend filter (price > HMA21 = uptrend, price < HMA21 = downtrend).
- Entry: Long when price breaks above H4 level AND price > 12h HMA21 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below L4 level AND price < 12h HMA21 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when price crosses below L4 level; Short exits when price crosses above H4 level.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide intraday support/resistance; HMA21 filters higher-timeframe trend with less lag than EMA; volume spike confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final WMA of raw HMA
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21
    close_12h = df_12h['close'].values
    hma_21 = calculate_hma(close_12h, 21)
    
    # Align HMA21 to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Calculate Williams Fractals on 12h for Camarilla levels (requires 2-bar confirmation delay)
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values,
    )
    # Bearish fractal = potential resistance, Bullish fractal = potential support
    # Need 2 extra bars for confirmation after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate Camarilla levels from previous 12h bar
    # H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # We'll use the previous completed 12h bar's values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_h4 = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_l4 = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Align Camarilla levels to 4h timeframe (these are based on completed 12h bars)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Get 4h data for volume MA(20)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21)  # volume MA needs 20, HMA21 needs 21
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_21_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter from 12h HMA21
        uptrend = curr_close > hma_21_aligned[i]
        downtrend = curr_close < hma_21_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above H4 level
                if curr_high > camarilla_h4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below L4 level
                if curr_low < camarilla_l4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below L4 level
            if curr_low < camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above H4 level
            if curr_high > camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_12hHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0