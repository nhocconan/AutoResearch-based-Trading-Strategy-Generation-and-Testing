#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend filter (price above/below weekly EMA) for strong regime alignment.
- Entry: Long when price breaks above Camarilla H3 AND price > 1w EMA50 AND volume > 2.0 * 12h volume MA(20);
         Short when price breaks below Camarilla L3 AND price < 1w EMA50 AND volume > 2.0 * 12h volume MA(20).
- Exit: ATR(14) trailing stop (long: highest_high - 2.5*ATR; short: lowest_low + 2.5*ATR).
- Signal size: 0.25 discrete to balance profit potential and fee drag.
- Designed to capture strong multi-day trends with volatility-adjusted exits and volume confirmation.
- Using 1w EMA50 instead of shorter MA for better trend definition and fewer whipsaws in bear markets.
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
    
    # Get 12h data for Camarilla and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels (H3, L3) on 12h using previous day's OHLC
    camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) / 4
    camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) / 4
    
    # Calculate ATR(14) on 12h for stoploss
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_12h[0] - low_12h[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track extreme prices for trailing stop
    highest_high = 0.0
    lowest_low = float('inf')
    
    # Start from index where all indicators are ready (max of 20 for volume, 14 for ATR, 50 for EMA)
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = float('inf')
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update trailing stop extremes
        if position == 1:  # long
            if curr_high > highest_high:
                highest_high = curr_high
        elif position == -1:  # short
            if curr_low < lowest_low:
                lowest_low = curr_low
        
        # Calculate stop levels
        long_stop = highest_high - 2.5 * atr_aligned[i] if highest_high > 0 else 0.0
        short_stop = lowest_low + 2.5 * atr_aligned[i] if lowest_low != float('inf') else float('inf')
        
        # Check for stoploss
        if position == 1 and curr_close < long_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        elif position == -1 and curr_close > short_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        
        # Breakout conditions
        bullish_breakout = curr_close > camarilla_h3_aligned[i]
        bearish_breakout = curr_close < camarilla_l3_aligned[i]
        
        # Trend filter from 1w EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 1w EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                    highest_high = curr_high
                    lowest_low = float('inf')
                # Short: bearish breakout AND price below 1w EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.25
                    position = -1
                    highest_high = 0.0
                    lowest_low = curr_low
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0