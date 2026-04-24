#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA21 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA21 trend filter to avoid whipsaws in ranging markets.
- Session filter: 08-20 UTC to trade only during active London/NY overlap.
- Entry: Long when price breaks above Camarilla H3 AND price > 4h EMA21 AND volume > 1.5 * 1h volume MA(20) AND session active;
         Short when price breaks below Camarilla L3 AND price < 4h EMA21 AND volume > 1.5 * 1h volume MA(20) AND session active.
- Exit: Close-based reversal (opposite signal) or stoploss via ATR trailing (implemented as signal=0 when price closes below/above Camarilla pivot point).
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday structure; 4h EMA21 filters counter-trend breakouts in bear markets like 2025.
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
    
    # Pre-compute session hours (08-20 UTC) for performance
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 1h timeframe (already daily, so just forward fill)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 4h data for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA(21) on 4h
    ema_21 = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA21 to 1h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21)
    
    # Calculate volume MA(20) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21)  # volume MA(20) and EMA21
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_session = in_session[i]
        
        # Stoploss: exit if price closes below/above Camarilla pivot point
        if position == 1:
            if curr_close < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation, trend filter, and session filter
        bullish_breakout = curr_close > camarilla_h3_aligned[i]
        bearish_breakout = curr_close < camarilla_l3_aligned[i]
        
        # Trend filter from 4h EMA21
        price_above_ema = curr_close > ema_21_aligned[i]
        price_below_ema = curr_close < ema_21_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        # Session filter
        session_ok = curr_session
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and session_ok:
                # Long: bullish breakout AND price above 4h EMA21
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.20
                    position = 1
                # Short: bearish breakout AND price below 4h EMA21
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA21_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0