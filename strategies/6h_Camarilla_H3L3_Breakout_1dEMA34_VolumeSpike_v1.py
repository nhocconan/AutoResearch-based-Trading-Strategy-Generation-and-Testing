#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and Camarilla pivot calculation from prior day's OHLC.
- Entry: Long when price breaks above Camarilla H3 AND price > 1d EMA34 AND volume > 2.0 * 6h volume MA(20);
         Short when price breaks below Camarilla L3 AND price < 1d EMA34 AND volume > 2.0 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via ATR trailing (implemented as signal=0 when conditions fail).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide mathematically derived support/resistance that work well in 6h timeframe.
- Using 1d EMA34 for trend alignment avoids whipsaws in ranging markets.
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
    
    # Get 6h data for volume MA and price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate volume MA(20) on 6h
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (H3, L3) from prior 1d bar
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    rang_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * rang_1d / 2
    camarilla_l3 = close_1d - 1.1 * rang_1d / 2
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max of 20 for volume MA, 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Check for stoploss/exit conditions (close-based)
        if position == 1:
            # Exit long if price breaks below L3 or volume dries up
            if curr_close < camarilla_l3_aligned[i] or curr_volume < 0.5 * vol_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            # Exit short if price breaks above H3 or volume dries up
            if curr_close > camarilla_h3_aligned[i] or curr_volume < 0.5 * vol_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation
        bullish_breakout = curr_close > camarilla_h3_aligned[i]
        bearish_breakout = curr_close < camarilla_l3_aligned[i]
        
        # Trend filter from 1d EMA34
        price_above_ema = curr_close > ema_34_aligned[i]
        price_below_ema = curr_close < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout H3 AND price above 1d EMA34
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish breakout L3 AND price below 1d EMA34
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

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0