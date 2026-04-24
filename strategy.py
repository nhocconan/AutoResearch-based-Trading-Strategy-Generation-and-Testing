#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter and Camarilla levels (more stable for BTC/ETH structure).
- Entry: Long when price breaks above H3 AND close > 4h EMA50 AND volume > 1.5 * 1h volume MA(20);
         Short when price breaks below L3 AND close < 4h EMA50 AND volume > 1.5 * 1h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via trend filter (signal=0 when price closes below/above 4h EMA50).
- Signal size: 0.20 discrete to minimize fee drag.
- Camarilla levels provide intraday support/resistance; 4h EMA50 filters counter-trend noise.
- Volume confirmation avoids false breakouts. Works in both bull and bear markets by following the 4h trend.
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
    
    # Get 4h data for Camarilla levels and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate EMA(50) on 4h close
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels (H3, L3) on 4h
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Get 1h data for volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators to primary 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above 4h EMA50 (trend filter)
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
        
        # Entry conditions with volume confirmation and Camarilla breakout
        bullish_breakout = curr_high > camarilla_h3_aligned[i]
        bearish_breakout = curr_low < camarilla_l3_aligned[i]
        
        # Trend filter from 4h EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above H3 AND close above 4h EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below L3 AND close below 4h EMA50
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

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0