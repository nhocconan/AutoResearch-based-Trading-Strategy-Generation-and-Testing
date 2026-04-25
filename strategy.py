#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeRegime_ATRStop
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter, volume regime filter (current volume > 1.5x 50-bar average), and ATR-based stoploss. Uses discrete sizing (0.30) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year. The R3/S3 levels provide stronger breakout signals, while the 12h EMA50 and volume regime filter reduce false signals in choppy markets. Works in both bull and bear by following the 12h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14 periods)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: current volume > 1.5x 50-period average (regime filter)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_regime = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need 50-period data for volume MA and 50 for 12h EMA
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Calculate Camarilla levels for current 4h bar using previous day's OHLC
        # We need to get the previous day's OHLC from 1d data
        # Since we don't have 1d data directly, we'll approximate using available data
        # In practice, we would use 1d data, but for this implementation we use a proxy
        # We'll use the 12h data to estimate daily levels (2x 12h bars = 1 day)
        if i >= 2:  # Need at least 2 previous 12h bars to form a day
            # Get the OHLC for the synthetic day (last two 12h bars)
            prev_high_12h_1 = high_12h[i-1] if i-1 >= 0 else high_12h[0]
            prev_low_12h_1 = low_12h[i-1] if i-1 >= 0 else low_12h[0]
            prev_close_12h_1 = close_12h[i-1] if i-1 >= 0 else close_12h[0]
            
            prev_high_12h_2 = high_12h[i-2] if i-2 >= 0 else high_12h[0]
            prev_low_12h_2 = low_12h[i-2] if i-2 >= 0 else low_12h[0]
            prev_close_12h_2 = close_12h[i-2] if i-2 >= 0 else close_12h[0]
            
            # Synthetic daily OHLC (approximation)
            synth_high = max(prev_high_12h_1, prev_high_12h_2)
            synth_low = min(prev_low_12h_1, prev_low_12h_2)
            synth_close = (prev_close_12h_1 + prev_close_12h_2) / 2
        else:
            # Fallback for early bars
            synth_high = high_12h[0] if len(high_12h) > 0 else high[i]
            synth_low = low_12h[0] if len(low_12h) > 0 else low[i]
            synth_close = close_12h[0] if len(close_12h) > 0 else close[i]
        
        # Camarilla levels calculation (using synthetic daily OHLC)
        range_1d = synth_high - synth_low
        camarilla_r3 = synth_close + (range_1d * 1.1 / 4)  # R3 level
        camarilla_s3 = synth_close - (range_1d * 1.1 / 4)  # S3 level
        
        if position == 0:
            # Long: price breaks above R3 in 12h uptrend with volume regime
            bullish_breakout = (curr_close > camarilla_r3) and \
                              (close_12h[i] > ema_50_12h_aligned[i]) and \
                              volume_regime[i]
            # Short: price breaks below S3 in 12h downtrend with volume regime
            bearish_breakout = (curr_close < camarilla_s3) and \
                              (close_12h[i] < ema_50_12h_aligned[i]) and \
                              volume_regime[i]
            
            if bullish_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (2.0 * atr[i])
            elif bearish_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price breaks below S3 OR stoploss hit OR trend turns down
            if (curr_close < camarilla_s3) or \
               (curr_close < atr_stop) or \
               (close_12h[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price breaks above R3 OR stoploss hit OR trend turns up
            if (curr_close > camarilla_r3) or \
               (curr_close > atr_stop) or \
               (close_12h[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeRegime_ATRStop"
timeframe = "4h"
leverage = 1.0