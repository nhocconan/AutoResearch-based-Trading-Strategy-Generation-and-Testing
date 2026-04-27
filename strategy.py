#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v5
Hypothesis: Camarilla R3/S3 breakouts aligned with 1d EMA34 trend and volume spikes capture high-probability moves in both bull and bear markets. 
Added: Weekly trend filter (price vs 1w EMA50) avoids counter-trend trades. ATR-based stoploss controls risk. 
Key improvements: Reduced volume threshold to 1.5x (from 2.0x) to increase trade frequency while maintaining quality, 
and added minimum holding period of 3 bars to reduce churn. Discrete sizing (0.30) balances return and fee drag. 
Target: 75-200 total trades over 4 years.
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
    
    # Volume confirmation: current volume > 1.5 * 20-period average (reduced from 2.0x)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Choppiness Index regime filter (avoid breakouts in ranging markets)
    # CHOP(14) = 100 * log10(sum(TR(14)) / (ATR(14) * 14)) / log10(14)
    # CHOP > 61.8 = ranging market (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first bar
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only allow breakouts when not strongly ranging
    
    # Align all indicators to primary timeframe (4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)  # volume is LTF, but confirm using 1d avg
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)  # align chop filter from 1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.30   # Position size: 30% of capital (discrete level)
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    # Warmup: need Camarilla (1), EMA34 (34), EMA50 (50), volume avg (20), chop (14)
    start_idx = max(1, 34, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        close_val = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema34 = ema34_1d_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        chop_ok = chop_filter_aligned[i]
        
        if position == 0:
            bars_since_entry = 0
            # Determine trend alignment: price vs EMA34 (1d) and EMA50 (1w)
            uptrend = close_val > ema34 and close_val > ema50
            downtrend = close_val < ema34 and close_val < ema50
            
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
        elif position != 0:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars
            if bars_since_entry < 3:
                signals[i] = size * position
                continue
                
            # Exit conditions: stoploss (2.5*ATR) or Camarilla opposite level touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            
            if position == 1:
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

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v5"
timeframe = "4h"
leverage = 1.0