#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channel: 20-period high/low from prior day.
- Entry: Long when price breaks above upper band AND 1w EMA50 bullish AND volume > 1.5 * volume MA(20).
         Short when price breaks below lower band AND 1w EMA50 bearish AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below lower band,
        exit short when price crosses above upper band.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via trend filter and mean-reversion exits.
Focus on BTC and ETH as primary symbols.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian bands (20-period) from prior day OHLC
    # We need to calculate these on the primary timeframe (1d) data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Upper band: 20-period high, lower band: 20-period low
    upper_band = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_band = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 20)  # Need enough bars for EMA50, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above upper band AND 1w EMA50 bullish AND volume confirmed
            if curr_close > upper_band[i] and curr_close > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band AND 1w EMA50 bearish AND volume confirmed
            elif curr_close < lower_band[i] and curr_close < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below lower band (reversion to mean)
            if curr_close < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above upper band (reversion to mean)
            if curr_close > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0