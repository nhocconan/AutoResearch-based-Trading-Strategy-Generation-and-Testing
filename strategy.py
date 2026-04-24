#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and 1d volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA from 1d to avoid false breakouts.
- Entry: Long when price breaks above Camarilla R3 AND 1d EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla S3 AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout (R3/S3) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Why should work in bull/bear: Camarilla levels provide adaptive support/resistance; EMA34 filter ensures trend alignment; volume spike confirms institutional participation; discrete sizing controls drawdown in crashes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # Camarilla R3 = close_prev + (high_prev - low_prev) * 1.1/2
    # Camarilla S3 = close_prev - (high_prev - low_prev) * 1.1/2
    # We'll use 4h bars but calculate based on daily logic - simpler: use 20-bar lookback for proxy
    # Better: use actual 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d data
    d_close = df_1d['close'].values
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Camarilla R3 and S3 for each 1d bar
    camarilla_r3 = d_close + (d_high - d_low) * 1.1 / 2
    camarilla_s3 = d_close - (d_high - d_low) * 1.1 / 2
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 1d bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above Camarilla R3 AND 1d EMA34 bullish (price > EMA34)
                if curr_high > camarilla_r3_val and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Camarilla S3 AND 1d EMA34 bearish (price < EMA34)
                elif curr_low < camarilla_s3_val and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR loss of volume confirmation
            if curr_low < camarilla_s3_val or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR loss of volume confirmation
            if curr_high > camarilla_r3_val or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0