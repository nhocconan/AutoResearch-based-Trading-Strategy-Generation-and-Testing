#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
Long when price breaks above 4h Camarilla R3 level AND 12h close > 12h EMA34 (uptrend) AND volume > 2.5x 20-period MA.
Short when price breaks below 4h Camarilla S3 level AND 12h close < 12h EMA34 (downtrend) AND volume > 2.5x 20-period MA.
Exit when price retouches 4h Camarilla H4/L4 levels (mean reversion) or 12h trend reverses.
Camarilla levels provide tight intraday support/resistance; 12h EMA34 filters counter-trend trades; volume confirmation reduces false breakouts.
Designed for low trade frequency (target: 20-40/year) to minimize fee drag and work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # For intraday, we use the previous 4h bar's high/low/close as proxy for daily
    # This approximates Camarilla calculation for 4h timeframe
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla levels calculation
    # R4 = close + ((high-low) * 1.5/2)
    # R3 = close + ((high-low) * 1.25/2)
    # R2 = close + ((high-low) * 1.166/2)
    # R1 = close + ((high-low) * 1.0833/2)
    # PP = (high + low + close) / 3
    # S1 = close - ((high-low) * 1.0833/2)
    # S2 = close - ((high-low) * 1.166/2)
    # S3 = close - ((high-low) * 1.25/2)
    # S4 = close - ((high-low) * 1.5/2)
    
    hl_range = prev_high - prev_low
    r3 = prev_close + (hl_range * 1.25 / 2)
    s3 = prev_close - (hl_range * 1.25 / 2)
    h4 = prev_close + (hl_range * 1.1 / 2)  # Between R3 and R4
    l4 = prev_close - (hl_range * 1.1 / 2)  # Between S3 and S4
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # Need at least 1 for shift, 34 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA34 = uptrend, close < EMA34 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_34_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_34_12h_aligned[i]
        
        # Volume filter: 4h volume > 2.5x 20-period MA (higher threshold to reduce trades)
        vol_filter = volume[i] > 2.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 level AND uptrend AND volume filter
            if close[i] > r3[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 level AND downtrend AND volume filter
            elif close[i] < s3[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retouches H4 level (mean reversion) OR 12h trend turns down
                if close[i] < h4[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retouches L4 level (mean reversion) OR 12h trend turns up
                if close[i] > l4[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0