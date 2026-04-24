#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter and Camarilla pivot levels (calculated from prior 1d bar).
- Entry: Long when price breaks above Camarilla H3 AND price > 1d EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla L3 AND price < 1d EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via ATR trailing (signal=0 when price closes below/above Camarilla midpoint).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday structure; 1d EMA50 filters counter-trend breakouts.
- Volume spike filter (2.0x MA) reduces false breakouts and overtrading.
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
    
    # Get 1d data for Camarilla calculation and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior 1d Camarilla levels (H3, L3, H4, L4, midpoint)
    # Use prior 1d bar's OHLC to avoid look-ahead
    camarilla_high = pd.Series(high_1d).shift(1)
    camarilla_low = pd.Series(low_1d).shift(1)
    camarilla_close = pd.Series(close_1d).shift(1)
    
    camarilla_range = camarilla_high - camarilla_low
    camarilla_h3 = camarilla_close + camarilla_range * 1.1 / 4
    camarilla_l3 = camarilla_close - camarilla_range * 1.1 / 4
    camarilla_h4 = camarilla_close + camarilla_range * 1.1 / 2
    camarilla_l4 = camarilla_close - camarilla_range * 1.1 / 2
    camarilla_mid = (camarilla_h3 + camarilla_l3) / 2
    
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max of 50 for EMA/Camarilla, 20 for vol MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above Camarilla midpoint
        if position == 1:
            if curr_close < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_close > camarilla_h3_aligned[i]
        bearish_breakout = curr_close < camarilla_l3_aligned[i]
        
        # Trend filter from 1d EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation (2.0x spike to reduce false breakouts)
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 1d EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish breakout AND price below 1d EMA50
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

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0