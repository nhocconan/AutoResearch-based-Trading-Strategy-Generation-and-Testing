#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Adaptive_v1
Hypothesis: Adaptive volume multiplier based on ATR regime prevents overtrading in low volatility.
Combines Camarilla R3/S3 breakouts with 1d trend alignment and volume confirmation scaled by volatility.
Target: 75-200 trades over 4 years by dynamically adjusting volume threshold.
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
    
    # Volatility-based volume threshold: ATR(34) scaled
    atr_34 = pd.Series(high - low).rolling(window=34, min_periods=34).mean().values
    vol_base = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    # Dynamic volume threshold: 1.5 + ATR/price ratio (scales with volatility)
    vol_threshold = 1.5 + (atr_34 / close) * 100  # Scales between 1.5-3.5 typically
    volume_confirm = volume > (vol_base * vol_threshold)
    
    # Choppiness Index regime filter (avoid breakouts in ranging markets)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only allow breakouts when not strongly ranging
    
    # Align all indicators to primary timeframe (4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.30   # Position size: 30% of capital
    
    # Warmup: need EMA34 (34), ATR (34), chop (14)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        chop_ok = chop_filter_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            if uptrend and vol_conf and chop_ok:
                # Long bias: long when price breaks above R3 with volume and not choppy
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf and chop_ok:
                # Short bias: short when price breaks below S3 with volume and not choppy
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.0*ATR) or Camarilla S3 touch
            atr_current = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_current
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s3:  # Camarilla S3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.0*ATR) or Camarilla R3 touch
            atr_current = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_current
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r3:  # Camarilla R3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Adaptive_v1"
timeframe = "4h"
leverage = 1.0