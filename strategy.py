#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v2
Hypothesis: Camarilla R3/S3 breakouts aligned with 1d EMA34 trend and volume spikes capture high-probability moves. Added EMA crossover filter on 4h to reduce whipsaws and improve trade quality. Uses discrete sizing (0.30) and ATR-based stoploss. Target: 75-200 total trades over 4 years.
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
    
    # Get 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (R3, S3) from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.125 * range_1d
    camarilla_s3 = close_1d - 1.125 * range_1d
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 4h EMA crossover filter (fast EMA12, slow EMA26)
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_crossover = ema12 > ema26  # bullish when fast > slow
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all indicators to primary timeframe (4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.30   # Position size: 30% of capital (discrete level)
    
    # Warmup: need Camarilla (1), EMA34 (34), EMA12 (12), EMA26 (26), volume avg (20)
    start_idx = max(1, 34, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema34 = ema34_1d_aligned[i]
        ema12_val = ema12[i]
        ema26_val = ema26[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA34 (1d) and EMA crossover (4h)
            uptrend = close_val > ema34 and ema12_val > ema26_val
            downtrend = close_val < ema34 and ema12_val < ema26_val
            
            if uptrend:
                # Long bias: long when price breaks above R3 with volume
                if (close_val > r3) and vol_conf:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend:
                # Short bias: short when price breaks below S3 with volume
                if (close_val < s3) and vol_conf:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.5*ATR) or Camarilla S3 touch or trend change
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.5 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s3:  # Camarilla S3 touch
                signals[i] = 0.0
                position = 0
            elif ema12_val <= ema26_val:  # trend change (EMA crossover down)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.5*ATR) or Camarilla R3 touch or trend change
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.5 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r3:  # Camarilla R3 touch
                signals[i] = 0.0
                position = 0
            elif ema12_val >= ema26_val:  # trend change (EMA crossover up)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0