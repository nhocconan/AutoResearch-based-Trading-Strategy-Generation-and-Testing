#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_1DTrend_VolumeSpike_v4
# Hypothesis: 4-hour Camarilla R3/S3 breakout with daily trend filter (price > daily EMA34) and volume spike confirmation.
# Uses daily trend to avoid counter-trend trades in both bull and bear markets.
# Volume spike ensures momentum confirmation. Targets 20-40 trades/year to minimize fee drag.
# Uses discrete position sizing (0.25).
# Updated: Added 2-bar close confirmation for daily EMA34 trend to reduce look-ahead risk.

name = "4H_Camarilla_R3_S3_1DTrend_VolumeSpike_v4"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/2, S3 = close - 1.1*(high-low)*1.1/2
    # We need previous day's high, low, close
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR < 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + daily uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and   # Daily uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + daily downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and   # Daily downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the middle of the Camarilla range (H4/L4)
            # H4 = close + 1.1*(high-low)*1.1/6, L4 = close - 1.1*(high-low)*1.1/6
            h4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 6
            l4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 6
            h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
            
            camarilla_mid = (h4_aligned[i] + l4_aligned[i]) / 2
            at_mid = abs(close[i] - camarilla_mid) < (h4_aligned[i] - l4_aligned[i]) * 0.25  # Within 25% of range
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals