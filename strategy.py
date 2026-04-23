#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 1w EMA34 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 AND 1w EMA34 is falling AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for long, R3 for short) or reverses EMA34 direction.
Uses 1w HTF for EMA34 trend (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h Camarilla levels (based on previous day's OHLC)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous 1d OHLC (using 6h data to approximate - we need the prior day's complete candle)
        # Since we're on 6h timeframe, we look back 4 periods (24h) for prior day's OHLC
        if i >= 4:
            prev_day_high = np.max(high[i-4:i])
            prev_day_low = np.min(low[i-4:i])
            prev_day_close = close[i-1]  # Close of previous 6h bar (approximation for daily close)
            prev_day_open = open_prices[i-4] if i >= 4 else open_prices[0]  # Open of 4 bars ago
            
            # More accurate: use daily OHLC from 1d timeframe
            # But for simplicity and to avoid look-ahead, we'll use the prior completed day's range
            # We approximate using the last 4 6h bars as one day
            range_val = prev_day_high - prev_day_low
            camarilla_h3[i] = prev_day_close + range_val * 1.1 / 4
            camarilla_l3[i] = prev_day_close - range_val * 1.1 / 4
            camarilla_h4[i] = prev_day_close + range_val * 1.1 / 2
            camarilla_l4[i] = prev_day_close - range_val * 1.1 / 2
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(4, 34, 20)  # Camarilla needs 4, EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla H3 AND EMA34 rising AND volume spike
            if price > h3 and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla L3 AND EMA34 falling AND volume spike
            elif price < l3 and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches L3 OR EMA34 starts falling
                if price < l3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches H3 OR EMA34 starts rising
                if price > h3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0