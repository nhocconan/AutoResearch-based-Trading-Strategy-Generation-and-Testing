#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla R3/S3 levels act as intraday support/resistance; breakouts with volume and 1d EMA34 trend filter capture strong moves. Chop filter avoids whipsaws in ranging markets. Designed for 4h timeframe to target 19-50 trades/year (75-200 over 4 years), minimizing fee drag. Works in bull markets via long breakouts and bear markets via short breakouts, with trend alignment preventing counter-trend entries.
"""

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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    range_ = prev_high - prev_low
    
    # Camarilla R3, S3 levels
    camarilla_r3 = prev_close + (range_ * 1.1 / 4)
    camarilla_s3 = prev_close - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (1d -> 4h: 6 bars per day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter: avoid ranging markets
    # CHOP(14) = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # We use a simplified version: if price is within BB(20,2) and ADX < 20, it's choppy
    # Instead, we use: if price is near VWAP and volatility is low -> choppy
    # Practical approximation: if price is within 1.5 * ATR(14) of VWAP-like value -> choppy
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    vwap_approx = (high + low + close) / 3  # Typical price as VWAP proxy
    dev_from_vwap = np.abs(close - vwap_approx)
    choppy = dev_from_vwap < (1.5 * atr_14)  # Low deviation from average price = choppy
    not_choppy = ~choppy  # We want to trade when NOT choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(34, 20, 14)  # EMA, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_not_choppy = not_choppy[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla R3 AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > camarilla_r3_aligned[i]) and bullish_bias and vol_spike and is_not_choppy
            # Short: price breaks below Camarilla S3 AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < camarilla_s3_aligned[i]) and bearish_bias and vol_spike and is_not_choppy
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla S3 (mean reversion) OR loss of bullish bias
            if (curr_low < camarilla_s3_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla R3 (mean reversion) OR loss of bearish bias
            if (curr_high > camarilla_r3_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0