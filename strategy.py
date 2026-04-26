#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike captures strong momentum moves while avoiding false breakouts in chop. Works in bull/bear via 1d trend alignment. Uses discrete sizing (0.25) to reduce fee churn and target 20-50 trades/year. Adds ATR-based stoploss to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (using 1d OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25  # Reduced size to lower drawdown
    
    # Warmup: max of EMA34 (34), volume MA (20), ATR (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price breaks above R3 with 1d uptrend and volume spike
        long_condition = (close_val > r3_val) and uptrend and vol_spike
        # Short: price breaks below S3 with 1d downtrend and volume spike
        short_condition = (close_val < s3_val) and downtrend and vol_spike
        
        # Exit conditions
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            # Exit: price re-enters R3-S3 range OR ATR stoploss hit
            if close_val < r3_val:
                long_exit = True
            elif close_val < entry_price - 2.0 * atr_val:  # ATR stoploss
                long_exit = True
        elif position == -1:  # Short position
            # Exit: price re-enters R3-S3 range OR ATR stoploss hit
            if close_val > s3_val:
                short_exit = True
            elif close_val > entry_price + 2.0 * atr_val:  # ATR stoploss
                short_exit = True
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0