#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND daily close > daily EMA34 AND volume > 1.3x average.
Short when price breaks below Camarilla S1 AND daily close < daily EMA34 AND volume > 1.3x average.
Exit when price touches the opposite Camarilla level (H3/L3) or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-25 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, daily trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 12h data for Camarilla levels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels on 12h data (based on previous 12h bar)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We use R1/S1 for entry, R3/S3 for exit
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = close_12h[0]
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    camarilla_hi = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 12  # R1
    camarilla_lo = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 12  # S1
    camarilla_hi3 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 4  # R3
    camarilla_lo3 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 4  # S3
    
    # Calculate ATR(14) on 12h data for stoploss
    tr1 = np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1)))
    tr2 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_12h[0] - low_12h[0]  # first bar
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_hi[i]) or np.isnan(camarilla_lo[i]) or 
            np.isnan(camarilla_hi3[i]) or np.isnan(camarilla_lo3[i]) or
            np.isnan(atr_12h[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 12h close for breakout conditions
        bar_close = close_12h[i]
        bar_high = high_12h[i]
        bar_low = low_12h[i]
        bar_volume = volume_12h[i]
        
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND daily uptrend AND volume confirmation
            if (bar_high > camarilla_hi[i] and 
                bar_close > ema34_1d_aligned[i] and 
                bar_volume > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = bar_close
            # Short: Price breaks below Camarilla S1 AND daily downtrend AND volume confirmation
            elif (bar_low < camarilla_lo[i] and 
                  bar_close < ema34_1d_aligned[i] and 
                  bar_volume > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = bar_close
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price touches Camarilla S3 (mean reversion) OR ATR stoploss
                if bar_low <= camarilla_lo3[i]:
                    exit_signal = True
                elif bar_close < entry_price - 2.5 * atr_12h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price touches Camarilla R3 (mean reversion) OR ATR stoploss
                if bar_high >= camarilla_hi3[i]:
                    exit_signal = True
                elif bar_close > entry_price + 2.5 * atr_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_1dEMA34_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0