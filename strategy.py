#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike_v10
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d ATR-based trend filter (price > EMA34 + 0.5*ATR for uptrend, < EMA34 - 0.5*ATR for downtrend) and volume spike (>2.0x median). This adds trend strength confirmation beyond simple EMA crossover to avoid whipsaws in ranging markets. Uses discrete position sizing (0.30) and ATR trailing stop (2.5x) for risk management. Designed for BTC/ETH with ~25-40 trades/year to avoid fee drag.
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
    
    # Get 1d data for HTF trend (EMA34 + ATR20)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d ATR(20) for trend strength filter
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_20_1d = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    cam_high = pd.Series(df_4h['high'].values).shift(1).values
    cam_low = pd.Series(df_4h['low'].values).shift(1).values
    cam_close = pd.Series(df_4h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Volume spike filter: volume > 2.0x median volume (30-period) for high conviction
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR(20) for volatility-based stops (on 4h)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, ATR(20) 1d, Camarilla (need 2 bars for shift), volume median (30), ATR (20)
    start_idx = max(34, 20, 2, 30, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_20_1d_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        atr_20_1d_val = atr_20_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter with strength: price > EMA34 + 0.5*ATR (uptrend) or < EMA34 - 0.5*ATR (downtrend)
        uptrend = close_val > (ema_34_1d_val + 0.5 * atr_20_1d_val)
        downtrend = close_val < (ema_34_1d_val - 0.5 * atr_20_1d_val)
        
        # Volume spike filter: only trade in high-volume environments
        volume_spike = volume_val > 2.0 * vol_median_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and strong uptrend
            long_signal = (close_val > r1_val) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S1 with volume spike, and strong downtrend
            short_signal = (close_val < s1_val) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike_v10"
timeframe = "4h"
leverage = 1.0