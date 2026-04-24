#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 100-200 total trades over 4 years (25-50/year).
- HTF: 12h EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above H3 level AND price > 12h EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below L3 level AND price < 12h EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR-based stoploss (2.0 * ATR(14)) and time-based exit (hold max 10 bars).
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels for precise intraday support/resistance, 12h EMA34 trend filter to avoid counter-trend trades,
  and volume confirmation for institutional participation. Designed to work in both bull and bear markets.
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
    
    # Get 4h data for Camarilla levels, ATR, and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels for 4h timeframe
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #          L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # But we only need H3 and L3 for breakout
    hl_range = high_4h - low_4h
    camarilla_h3 = close_4h + 1.1 * hl_range * 1.1 / 4
    camarilla_l3 = close_4h - 1.1 * hl_range * 1.1 / 4
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA needs 20, EMA34 needs 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(vol_ma_4h[i]) or 
            np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_since_entry = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Break above H3 level AND price > 12h EMA34 (uptrend)
                if curr_high > camarilla_h3[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    bars_since_entry = 0
                # Short: Break below L3 level AND price < 12h EMA34 (downtrend)
                elif curr_low < camarilla_l3[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    bars_since_entry = 0
        elif position != 0:
            # Update bars since entry
            bars_since_entry += 1
            
            # Check exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Stoploss: 2.0 * ATR below entry
                stoploss = entry_price - 2.0 * curr_atr
                # Time-based exit: hold max 10 bars
                if curr_close < stoploss or bars_since_entry >= 10:
                    exit_signal = True
            elif position == -1:  # Short position
                # Stoploss: 2.0 * ATR above entry
                stoploss = entry_price + 2.0 * curr_atr
                # Time-based exit: hold max 10 bars
                if curr_close > stoploss or bars_since_entry >= 10:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0