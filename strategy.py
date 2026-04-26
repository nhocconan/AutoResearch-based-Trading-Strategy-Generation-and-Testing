#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v3
Hypothesis: Trade 4h breakouts from tighter Camarilla R1/S1 levels with 1d EMA34 trend filter and volume confirmation. Uses 2-bar breakout confirmation to reduce false signals. Targets 30-50 trades/year on BTC/ETH. Works in bull/bear via trend filter; Camarilla levels provide structure in ranging markets. Discrete size 0.30 limits fee drag. Added ATR-based volatility filter to avoid choppy markets.
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
    
    # Get 1d data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla pivot levels (R1, S1) from previous 1d bar
    # Formula: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # first bar has no previous
    
    # Calculate Camarilla R1 and S1 for previous 1d bar (tighter levels)
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 12
    
    # 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 4h ATR(14) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d data to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Volatility filter: ATR > 20-period ATR mean (avoid extremely low volatility)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > 0.5 * atr_ma  # Allow low volatility but not extreme
    
    # Require close to stay beyond level for 2 consecutive bars to reduce false breakouts
    close_above_r1 = close > camarilla_r1_aligned
    close_below_s1 = close < camarilla_s1_aligned
    close_above_r1_2bar = close_above_r1 & np.roll(close_above_r1, 1)
    close_below_s1_2bar = close_below_s1 & np.roll(close_below_s1, 1)
    close_above_r1_2bar[0] = False
    close_below_s1_2bar[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), volume MA (20), ATR (14+20)
    start_idx = max(34, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + 1d uptrend + 2-bar confirmation + vol filter
            long_breakout = close_above_r1_2bar[i]
            long_signal = long_breakout and volume_spike[i] and trend_1d_uptrend and vol_filter[i]
            
            # Short: price breaks below S1 + volume spike + 1d downtrend + 2-bar confirmation + vol filter
            short_breakout = close_below_s1_2bar[i]
            short_signal = short_breakout and volume_spike[i] and trend_1d_downtrend and vol_filter[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price touches S1 level OR 1d trend turns down
            if (close[i] < camarilla_s1_aligned[i] or not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price touches R1 level OR 1d trend turns up
            if (close[i] > camarilla_r1_aligned[i] or not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0