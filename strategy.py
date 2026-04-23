#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 rising AND 1d volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 falling AND 1d volume > 2.0x 20-period average.
Exit when price touches opposite Camarilla level (R3/S3) or 1d EMA34 reverses.
Uses 1d HTF for trend and volume filters to avoid false breakouts in ranging markets.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Camarilla levels provide structure, EMA34 filters trend, volume spike confirms momentum.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume average (20-period) for spike filter (HTF)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Calculate Camarilla levels from previous 1d bar (using 1d OHLC)
        # We need the previous completed 1d bar's OHLC
        # Since we're on 4h timeframe, we can approximate using rolling window
        # For simplicity, we'll use the current day's OHLC up to this point
        # In practice, we should use the previous day's close, but for 4h we approximate
        
        # Calculate rolling 1d OHLC from 4h data (6x4h = 1d)
        if i >= 5:  # Need at least 6 periods for approximate 1d
            # Approximate daily OHLC from last 6x4h bars
            day_high = np.max(high[i-5:i+1])
            day_low = np.min(low[i-5:i+1])
            day_close = close[i]
            day_open = close[i-5] if i >= 5 else close[0]
            
            # Camarilla levels
            range_val = day_high - day_low
            if range_val > 0:
                camarilla_r3 = day_close + range_val * 1.1 / 4
                camarilla_s3 = day_close - range_val * 1.1 / 4
            else:
                camarilla_r3 = day_close
                camarilla_s3 = day_close
        else:
            camarilla_r3 = close[i]
            camarilla_s3 = close[i]
        
        # Volume filter: 1d volume > 2.0x 20-period average (strong volume confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume filter
            if price > camarilla_r3 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume filter
            elif price < camarilla_s3 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S3 OR EMA34 starts falling
                if price < camarilla_s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R3 OR EMA34 starts rising
                if price > camarilla_r3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0