#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter and volume confirmation (>2x 20-bar avg) to capture intraday momentum while avoiding false breakouts. Uses 4h trend for signal direction (proven edge) and 1h only for precise entry timing to limit trades to 15-37/year. Designed for BTC/ETH in both bull/bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA50 on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 4h data (based on previous bar's OHLC)
    camarilla_r1_4h = close_4h + ((high_4h - low_4h) * 1.1 / 12)
    camarilla_s1_4h = close_4h - ((high_4h - low_4h) * 1.1 / 12)
    camarilla_h4_4h = close_4h + ((high_4h - low_4h) * 1.1 / 2)
    camarilla_l4_4h = close_4h - ((high_4h - low_4h) * 1.1 / 2)
    
    # Align HTF indicators to 1h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4_4h, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4_4h, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume (strict filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 4h trend with volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0