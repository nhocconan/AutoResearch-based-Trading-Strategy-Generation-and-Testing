#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d HMA21 trend filter and volume spike confirmation.
- Primary timeframe: 4h to target 75-200 total trades over 4 years (19-50/year).
- HTF: 1d HMA21 for trend direction (bullish if close > HMA21, bearish if close < HMA21).
- Camarilla levels: H3 and L3 from prior 1d session (using prior close to avoid look-ahead).
- Entry: Long when price breaks above prior H3 AND 1d HMA21 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior L3 AND 1d HMA21 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 1d HMA21,
        exit short when price crosses above 1d HMA21.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy targets institutional pivot levels with trend and volume confirmation,
designed to work in both bull and bear markets by aligning with the 1d trend.
HMA provides smoother trend with less lag than EMA, improving signal quality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(arr).ewm(span=half_period, adjust=False, min_periods=half_period).mean().values
    # WMA of full period
    wma_full = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA21 for trend filter
    df_1d_close = df_1d['close'].values
    hma_1d = calculate_hma(df_1d_close, 21)
    
    # Calculate prior 1d Camarilla H3 and L3 levels
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_vals = typical_price.values
    range_ = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla H3 = close + (high - low) * 1.1/2
    # Camarilla L3 = close - (high - low) * 1.1/2
    camarilla_h3 = df_1d['close'].values + range_ * 1.1 / 2
    camarilla_l3 = df_1d['close'].values - range_ * 1.1 / 2
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30)  # Need enough bars for HMA21 and Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior Camarilla H3 AND 1d HMA21 bullish AND volume confirmed
            if curr_close > camarilla_h3_aligned[i] and curr_close > hma_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior Camarilla L3 AND 1d HMA21 bearish AND volume confirmed
            elif curr_close < camarilla_l3_aligned[i] and curr_close < hma_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 1d HMA21 (trend change)
            if curr_close < hma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 1d HMA21 (trend change)
            if curr_close > hma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0