#!/usr/bin/env python3
# 4h_atr_breakout_1d_trend_volume_v3
# Hypothesis: On 4h timeframe, break out of ATR-based channel with volume confirmation and 1d trend filter.
# Long when price closes above ATR(10) upper band with volume > 1.5x average and 1d uptrend.
# Short when price closes below ATR(10) lower band with volume > 1.5x average and 1d downtrend.
# Exit when price crosses back through the opposite ATR band.
# Uses tighter entry conditions (1.5x volume) to reduce trade frequency and avoid fee drag.
# Target: 20-40 trades/year to stay well under the 400 total 4h trade limit.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v3"
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
    
    # ATR calculation (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR-based channels (1.5 * ATR)
    atr_mult = 1.5
    upper_channel = close + atr_mult * atr
    lower_channel = close - atr_mult * atr
    
    # 1d trend filter: EMA20
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_4h = align_htf_to_ltf(prices, df_daily, daily_ema20)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or \
           np.isnan(daily_ema20_4h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below lower channel
            if close[i] < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above upper channel
            if close[i] > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema20_4h[i]
            daily_downtrend = close[i] < daily_ema20_4h[i]
            
            # Long entry: price closes above upper channel with volume and uptrend
            if close[i] > upper_channel[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below lower channel with volume and downtrend
            elif close[i] < lower_channel[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals