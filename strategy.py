#!/usr/bin/env python3
"""
4h_keltner_breakout_12h_trend_volume_v1
Hypothesis: On 4h timeframe, use 12h EMA for trend filter and Keltner Channel (ATR-based) for breakout signals, with volume confirmation to filter weak moves. Enter long when price breaks above upper Keltner band with price above EMA and volume confirmation; enter short when price breaks below lower Keltner band with price below EMA and volume confirmation. Exit when price crosses back over EMA or reverses across Keltner middle band. This strategy captures strong trending moves with volatility-adjusted breakouts, reducing false signals in choppy markets. Works in both bull and bear via trend filter and volatility-based bands.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_keltner_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter and ATR for Keltner
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # 12h ATR for Keltner Channel (using 20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr_12h = pd.Series(tr).ewm(alpha=1/20, adjust=False).mean().values
    
    # Keltner Channel components (20-period EMA of typical price)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    keltner_middle = pd.Series(typical_price_12h).ewm(span=20, adjust=False).mean().values
    keltner_upper = keltner_middle + 2.0 * atr_12h
    keltner_lower = keltner_middle - 2.0 * atr_12h
    
    # Align indicators to 4h timeframe
    ema_12h_4h = align_htf_to_ltf(prices, df_12h, ema_12h)
    keltner_middle_4h = align_htf_to_ltf(prices, df_12h, keltner_middle)
    keltner_upper_4h = align_htf_to_ltf(prices, df_12h, keltner_upper)
    keltner_lower_4h = align_htf_to_ltf(prices, df_12h, keltner_lower)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_4h[i]) or np.isnan(keltner_middle_4h[i]) or
            np.isnan(keltner_upper_4h[i]) or np.isnan(keltner_lower_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_12h_4h[i]
        downtrend = close[i] < ema_12h_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below EMA (trend change)
            if close[i] < ema_12h_4h[i]:
                exit_long = True
            # Exit if price crosses below middle Keltner band (momentum loss)
            elif close[i] < keltner_middle_4h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price crosses above EMA (trend change)
            if close[i] > ema_12h_4h[i]:
                exit_short = True
            # Exit if price crosses above middle Keltner band (momentum loss)
            elif close[i] > keltner_middle_4h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above upper Keltner band with uptrend and volume confirmation
            if close[i] > keltner_upper_4h[i] and close[i-1] <= keltner_upper_4h[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below lower Keltner band with downtrend and volume confirmation
            if close[i] < keltner_lower_4h[i] and close[i-1] >= keltner_lower_4h[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals