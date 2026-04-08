#!/usr/bin/env python3
# 4h_atr_breakout_12h_trend_volume_v1
# Hypothesis: On 4h timeframe, use ATR-based volatility breakout with 12h trend filter and volume confirmation.
# Long when price breaks above ATR(14) upper band with volume > 1.5x average and 12h uptrend.
# Short when price breaks below ATR(14) lower band with volume > 1.5x average and 12h downtrend.
# Exit when price reverses by 1x ATR from breakout level or opposite signal triggers.
# Uses volatility breakout to capture momentum moves with volatility-adjusted position sizing.
# Target: 25-35 trades/year to avoid excessive fee drag while capturing significant moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility bands
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros(n)
    atr[13] = np.mean(tr[:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volatility bands: midpoint ± ATR
    midpoint = (high + low) / 2
    upper_band = midpoint + atr
    lower_band = midpoint - atr
    
    # Get 12h trend data (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA25 for trend filter
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, min_periods=25, adjust=False).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    breakout_level = 0  # Track breakout level for exit
    
    # Start after warmup
    start_idx = 25
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or \
           np.isnan(ema25_12h_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverses by 1x ATR from breakout level OR opposite signal
            if close[i] <= breakout_level - atr[i] or \
               (close[i] < lower_band[i] and volume[i] > 1.5 * avg_volume[i] and close[i] < ema25_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverses by 1x ATR from breakout level OR opposite signal
            if close[i] >= breakout_level + atr[i] or \
               (close[i] > upper_band[i] and volume[i] > 1.5 * avg_volume[i] and close[i] > ema25_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # 12h trend filter
            uptrend_12h = close[i] > ema25_12h_aligned[i]
            downtrend_12h = close[i] < ema25_12h_aligned[i]
            
            # Long entry: price breaks above upper band with volume and uptrend
            if close[i] > upper_band[i] and volume_ok and uptrend_12h:
                position = 1
                breakout_level = close[i]  # Record breakout level
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume and downtrend
            elif close[i] < lower_band[i] and volume_ok and downtrend_12h:
                position = -1
                breakout_level = close[i]  # Record breakout level
                signals[i] = -0.25
    
    return signals