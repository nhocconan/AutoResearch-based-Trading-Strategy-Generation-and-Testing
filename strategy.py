#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_StopLoss_v1
Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation (>2.0x 20-bar average) captures strong trending moves with close-based stoploss (price crosses opposite Camarilla level). Uses discrete sizing (0.25) to target ~20-35 trades/year. Works in bull/bear by only taking breakouts aligned with 12h trend. Avoids look-ahead by using mtf_data for proper HTF alignment.
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # Already shifted for completed day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to LTF (they update only when new 1d bar completes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ATR(14) for volatility filter (optional)
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), Camarilla needs 1d data, ATR (14), volume MA (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        trend_val = ema50_12h_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 12h EMA50 = uptrend, price < 12h EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Camarilla breakout in direction of 12h trend + volume
        long_entry = close_val > r1_val and is_uptrend and vol_conf
        short_entry = close_val < s1_val and is_downtrend and vol_conf
        
        # Exit conditions: price crosses opposite Camarilla level (mean reversion) or ATR stop
        long_exit = False
        short_exit = False
        if position == 1:
            # Long exit: price crosses below S1 (opposite level) or ATR stop
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < s1_val or close_val < stop_price
        elif position == -1:
            # Short exit: price crosses above R1 (opposite level) or ATR stop
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > r1_val or close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Approximate entry price for stop calculation
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_StopLoss_v1"
timeframe = "4h"
leverage = 1.0