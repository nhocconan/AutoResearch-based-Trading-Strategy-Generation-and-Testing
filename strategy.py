#!/usr/bin/env python3
"""
1h_4h_1d_SMC_OrderBlock_Breakout
Hypothesis: Uses 1d order blocks (OB) from institutional supply/demand zones as primary trend filter.
On 1h, enters long when price breaks above bullish OB with 4h bullish trend and volume spike.
Enters short when price breaks below bearish OB with 4h bearish trend and volume spike.
Exits when price returns to opposite OB or 4h trend flips.
Designed for low trade frequency (15-35/year) by requiring confluence of 1d structure, 4h trend, and 1h breakout with volume.
Works in bull/bear markets by following institutional order flow on higher timeframes.
"""

name = "1h_4h_1d_SMC_OrderBlock_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def find_order_blocks(high, low, close, lookback=20):
    """
    Find bullish and bearish order blocks.
    Bullish OB: last down candle before a strong up move (close > previous high)
    Bearish OB: last up candle before a strong down move (close < previous low)
    Returns arrays of OB high and low levels (np.nan if no OB)
    """
    n = len(close)
    ob_high = np.full(n, np.nan)
    ob_low = np.full(n, np.nan)
    
    for i in range(2, n):
        # Bullish OB: candle i-2 is down, i-1 is up, and close[i] > high[i-1]
        if close[i-2] < open_[i-2] and close[i-1] > open_[i-1] and close[i] > high[i-1]:
            ob_high[i] = high[i-2]
            ob_low[i] = low[i-2]
        # Bearish OB: candle i-2 is up, i-1 is down, and close[i] < low[i-1]
        elif close[i-2] > open_[i-2] and close[i-1] < open_[i-1] and close[i] < low[i-1]:
            ob_high[i] = high[i-2]
            ob_low[i] = low[i-2]
        # Propagate last valid OB forward
        else:
            if not np.isnan(ob_high[i-1]):
                ob_high[i] = ob_high[i-1]
                ob_low[i] = ob_low[i-1]
    
    return ob_high, ob_low

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract OHLCV
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Order Blocks for Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    ob_high_1d, ob_low_1d = find_order_blocks(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, df_1d['open'].values
    )
    
    # Align 1d OB to 1h
    ob_high_1d_1h = align_htf_to_ltf(prices, df_1d, ob_high_1d)
    ob_low_1d_1h = align_htf_to_ltf(prices, df_1d, ob_low_1d)
    
    # --- 4h Trend Filter (EMA34) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_1h = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # --- Volume Spike (20-period) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ob_high_1d_1h[i]) or np.isnan(ob_low_1d_1h[i]) or
            np.isnan(ema_34_4h_1h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price above 1d bullish OB, 4h bullish trend, breaks above OB with volume
            if (close[i] > ob_high_1d_1h[i] and  # above bullish OB
                close[i-1] <= ob_high_1d_1h[i-1] and  # broke above this bar
                ema_34_4h_1h[i] > ema_34_4h_1h[i-1] and  # 4h EMA rising
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: price below 1d bearish OB, 4h bearish trend, breaks below OB with volume
            elif (close[i] < ob_low_1d_1h[i] and  # below bearish OB
                  close[i-1] >= ob_low_1d_1h[i-1] and  # broke below this bar
                  ema_34_4h_1h[i] < ema_34_4h_1h[i-1] and  # 4h EMA falling
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to bearish OB or 4h trend turns bearish
                if (close[i] < ob_low_1d_1h[i] or 
                    ema_34_4h_1h[i] < ema_34_4h_1h[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to bullish OB or 4h trend turns bullish
                if (close[i] > ob_high_1d_1h[i] or 
                    ema_34_4h_1h[i] > ema_34_4h_1h[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals