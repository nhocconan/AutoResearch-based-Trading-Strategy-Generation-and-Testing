#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h trend filter (EMA34) and volume confirmation.
Long when price breaks above Camarilla R3 AND 4h EMA34 rising AND volume > 2.0x 24-period MA.
Short when price breaks below Camarilla S3 AND 4h EMA34 falling AND volume > 2.0x 24-period MA.
Exit when price touches opposite Camarilla level (R3/S3) or 4h EMA34 reverses.
Uses 4h HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Camarilla provides intraday support/resistance, 4h EMA34 filters major trend, volume confirms breakout strength.
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
    
    # Calculate 1h Camarilla levels (based on previous bar's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)  # for stop loss
    camarilla_s4 = np.full(n, np.nan)  # for stop loss
    
    for i in range(1, n):
        # Use previous bar's OHLC to avoid look-ahead
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        rng = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + rng * 1.1/2
        camarilla_s3[i] = prev_close - rng * 1.1/2
        camarilla_r4[i] = prev_close + rng * 1.1
        camarilla_s4[i] = prev_close - rng * 1.1
    
    # Calculate 4h EMA34 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h volume MA (24-period) for spike filter
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 24)  # Camarilla (needs 1), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i]) or
            hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_24[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1h volume > 2.0x 24-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume filter
            if price > r3 and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume filter
            elif price < s3 and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S3 (opposite) OR EMA34 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R3 (opposite) OR EMA34 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0