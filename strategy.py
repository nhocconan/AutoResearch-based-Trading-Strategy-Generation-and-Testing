#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with 1d trend filter and volume confirmation.
Long when price breaks above Camarilla R4 AND 1d close > 1d EMA34 AND volume > 1.5x 20-period average volume.
Short when price breaks below Camarilla S4 AND 1d close < 1d EMA34 AND volume > 1.5x 20-period average volume.
Exit when price retests Camarilla R3/S3 OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~12-30 trades/year on 6h timeframe.
Camarilla pivots from daily timeframe provide institutional support/resistance levels effective in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivots (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    #          R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # Use same close for calculation (will be aligned)
    
    # Calculate pivot levels
    rng = high_1d - low_1d
    r4_1d = close_1d_prev + 1.5 * rng
    s4_1d = close_1d_prev - 1.5 * rng
    r3_1d = close_1d_prev + 1.125 * rng
    s3_1d = close_1d_prev - 1.125 * rng
    
    # Align pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # vol_ma20, ema34_1d (pivots calculated from same data)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_34_1d_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R4 AND bullish trend AND volume spike
            if price > r4 and close[i] > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below S4 AND bearish trend AND volume spike
            elif price < s4 and close[i] < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retests R3/S3 (mean reversion at pivot support/resistance)
            if position == 1 and price < r3:
                exit_signal = True
            elif position == -1 and price > s3:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R4S4_Breakout_1dEMA34_Trend_VolumeSpike_R3S3Exit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0