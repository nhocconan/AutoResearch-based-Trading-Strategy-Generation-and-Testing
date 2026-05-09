#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Camarilla pivot levels from 1d OHLC to identify potential reversal or breakout points.
# Long when price breaks above R1 with volume and 1d EMA34 up; short when breaks below S1 with volume and 1d EMA34 down.
# Designed to capture institutional interest at key levels while filtering by higher timeframe trend.
# Camarilla levels are widely watched and provide objective entry/exit points with built-in risk management.
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L) * 1.1/12, S1 = C - (H-L) * 1.1/12
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Only calculate when we have previous day data
    valid_idx = ~(np.isnan(prev_close) | np.isnan(prev_high) | np.isnan(prev_low))
    R1 = np.full_like(prev_close, np.nan)
    S1 = np.full_like(prev_close, np.nan)
    R1[valid_idx] = prev_close[valid_idx] + (prev_high[valid_idx] - prev_low[valid_idx]) * 1.1 / 12
    S1[valid_idx] = prev_close[valid_idx] - (prev_high[valid_idx] - prev_low[valid_idx]) * 1.1 / 12
    
    # Camarilla levels need 2 extra bars for confirmation (after pivot forms)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1, additional_delay_bars=2)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1, additional_delay_bars=2)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume + 1d EMA34 up
            if (price > R1_aligned[i] and 
                vol_confirm[i] and 
                price > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume + 1d EMA34 down
            elif (price < S1_aligned[i] and 
                  vol_confirm[i] and 
                  price < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 or 1d EMA34 turns down
            if price < S1_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 or 1d EMA34 turns up
            if price > R1_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals