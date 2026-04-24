#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter (more stable than shorter periods).
- Entry: Long when price breaks above Camarilla H3 AND price > 12h EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla L3 AND price < 12h EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via trend filter (signal=0 when price closes below/above 12h EMA50).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Uses 12h EMA50 for better trend filtering in both bull and bear markets, reducing whipsaw.
- Camarilla levels provide intraday support/resistance; 12h EMA50 filters counter-trend breakouts.
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
    
    # Calculate typical price for Camarilla (using previous day's OHLC)
    typical_price = (high + low + close) / 3.0
    range_val = high - low
    
    # Calculate Camarilla levels (H3, L3) based on previous day's data
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    camarilla_h3 = close + 1.1 * range_val / 2.0
    camarilla_l3 = close - 1.1 * range_val / 2.0
    
    # Shift to use previous day's levels (lookback by 1 bar)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Get 12h data for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above 12h EMA50 (trend filter)
        if position == 1:
            if curr_close < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_close > camarilla_h3[i]
        bearish_breakout = curr_close < camarilla_l3[i]
        
        # Trend filter from 12h EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: breakout above H3 AND price above 12h EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: breakout below L3 AND price below 12h EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0