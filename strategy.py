#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v2
Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as key support/resistance. A break of these levels with volume spike (>2.0x 20-bar MA) and alignment with 1d EMA34 trend captures strong momentum moves. Added ATR-based trailing stop (3*ATR) to reduce drawdown in choppy markets. Discrete position sizing (0.25) limits fee drag. Works in bull markets (breakouts continuation) and bear markets (breakdown continuation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r1 = close + range_ * 1.1 / 12.0
    s1 = close - range_ * 1.1 / 12.0
    r2 = close + range_ * 1.1 / 6.0
    s2 = close - range_ * 1.1 / 6.0
    r3 = close + range_ * 1.1 / 4.0
    s3 = close - range_ * 1.1 / 4.0
    r4 = close + range_ * 1.1 / 2.0
    s4 = close - range_ * 1.1 / 2.0
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels and EMA34 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla levels
    _, r1_1d, _, _, _, s1_1d, _, _, _ = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (4h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # 4h ATR for trailing stop (3*ATR)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]]) if len(tr1) > 0 else 0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_high = 0.0
    min_low = 0.0
    
    # Start index: need volume MA (20), ATR (14), and aligned HTF arrays
    start_idx = max(20, 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and 1d uptrend
            long_breakout = (curr_close > r1_1d_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short: price breaks below S1 with volume spike and 1d downtrend
            short_breakout = (curr_close < s1_1d_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                max_high = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                min_low = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update max high for trailing stop
            max_high = max(max_high, curr_close)
            # Long exit: price breaks below S1, trend turns down, or ATR trailing stop hit
            trailing_stop = max_high - 3.0 * atr[i]
            if (curr_close < s1_1d_aligned[i]) or (curr_close < ema_34_1d_aligned[i]) or (curr_close < trailing_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update min low for trailing stop
            min_low = min(min_low, curr_close)
            # Short exit: price breaks above R1, trend turns up, or ATR trailing stop hit
            trailing_stop = min_low + 3.0 * atr[i]
            if (curr_close > r1_1d_aligned[i]) or (curr_close > ema_34_1d_aligned[i]) or (curr_close > trailing_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0