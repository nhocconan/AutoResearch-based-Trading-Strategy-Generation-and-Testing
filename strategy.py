#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_HTF_Filter_v5
Hypothesis: Camarilla R3/S3 breakouts aligned with 1d EMA34 trend and volume spikes capture high-probability moves. 
Added: 4h Supertrend filter to avoid whipsaws and ensure trend alignment. Weekly trend filter (price vs 1w EMA50) avoids counter-trend trades. 
ATR-based stoploss controls risk. Discrete sizing (0.30) balances return and fee drag. Target: 75-200 total trades over 4 years.
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
    
    # Get 1w data for weekly trend filter (price vs EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Supertrend filter (HTF: 4h)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = np.maximum(high_4h - low_4h, np.absolute(high_4h - np.roll(close_4h, 1)))
    tr1 = np.maximum(tr1, np.absolute(low_4h - np.roll(close_4h, 1)))
    tr1[0] = high_4h[0] - low_4h[0]  # first bar
    atr = pd.Series(tr1).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    upper_band = (high_4h + low_4h) / 2 + 3.0 * atr
    lower_band = (high_4h + low_4h) / 2 - 3.0 * atr
    
    supertrend = np.full(len(close_4h), np.nan, dtype=float)
    for i in range(1, len(close_4h)):
        if np.isnan(supertrend[i-1]):
            supertrend[i] = lower_band[i] if close_4h[i] > upper_band[i-1] else upper_band[i]
        else:
            if close_4h[i] <= supertrend[i-1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all indicators to primary timeframe (4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)  # volume is LTF, but confirm using 1d avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.30   # Position size: 30% of capital (discrete level)
    
    # Warmup: need Camarilla (1), EMA34 (34), EMA50 (50), volume avg (20), Supertrend (10)
    start_idx = max(1, 34, 50, 20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema34 = ema34_1d_aligned[i]
        ema50 = ema50_1w_aligned[i]
        supertrend_val = supertrend_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA34 (1d) and EMA50 (1w)
            uptrend = close_val > ema34 and close_val > ema50
            downtrend = close_val < ema34 and close_val < ema50
            
            # Supertrend filter: only long when price > Supertrend, short when price < Supertrend
            if uptrend and vol_conf and close_val > supertrend_val:
                # Long bias: long when price breaks above R3 with volume and trend alignment
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf and close_val < supertrend_val:
                # Short bias: short when price breaks below S3 with volume and trend alignment
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.5*ATR) or Camarilla S3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.5 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s3:  # Camarilla S3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.5*ATR) or Camarilla R3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.5 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r3:  # Camarilla R3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_HTF_Filter_v5"
timeframe = "4h"
leverage = 1.0