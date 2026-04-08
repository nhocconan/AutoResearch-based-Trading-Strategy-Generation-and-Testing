#!/usr/bin/env python3
# 6h_atr_breakout_1d_trend_volume_v1
# Hypothesis: On 6h timeframe, use ATR breakout with daily trend filter and volume confirmation.
# Long when price closes above ATR(14) upper band (mean + 2*ATR) with volume > 1.5x average and daily trend up.
# Short when price closes below ATR(14) lower band (mean - 2*ATR) with volume > 1.5x average and daily trend down.
# Exit on opposite ATR band touch or when volume drops below average.
# Daily trend defined by price above/below daily EMA20.
# ATR breakouts capture momentum bursts; daily trend filter avoids counter-trend trades; volume confirms strength.
# Designed for low trade frequency (12-37/year) to minimize fee drag while capturing significant moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_atr_breakout_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # ATR calculation on 6h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR bands: mean price +/- 2*ATR
    avg_price = (high + low + close) / 3
    atr_upper = avg_price + 2 * atr
    atr_lower = avg_price - 2 * atr
    
    # Daily trend filter: price above/below daily EMA20
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_6h = align_htf_to_ltf(prices, df_daily, daily_ema20)
    
    # Volume confirmation: 20-period average on 6h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(atr_upper[i]) or np.isnan(atr_lower[i]) or np.isnan(daily_ema20_6h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches ATR lower band (opposite) or volume drops below average
            if close[i] <= atr_lower[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches ATR upper band (opposite) or volume drops below average
            if close[i] >= atr_upper[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema20_6h[i]
            daily_downtrend = close[i] < daily_ema20_6h[i]
            
            # Long entry: price closes above ATR upper band with volume and uptrend
            if close[i] > atr_upper[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below ATR lower band with volume and downtrend
            elif close[i] < atr_lower[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals