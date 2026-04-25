#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout + 4h EMA50 Trend + Volume Spike + 1d ATR Stoploss
Hypothesis: Camarilla H3/L3 levels from 4h chart provide institutional breakout levels.
Breakouts in direction of 4h EMA50 trend with volume confirmation capture strong moves.
1d ATR-based stoploss limits drawdown during sideways/choppy periods.
Designed for 1h timeframe targeting 15-37 trades/year. Uses 4h for signal direction,
1h only for entry timing. Session filter (08-20 UTC) reduces noise trades.
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
    
    # Get 4h data for Camarilla pivots (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from 4h OHLC
    daily_high = df_4h['high'].values
    daily_low = df_4h['low'].values
    daily_close = df_4h['close'].values
    camarilla_h3 = daily_close + 1.1 * (daily_high - daily_low) / 4
    camarilla_l3 = daily_close - 1.1 * (daily_high - daily_low) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for ATR(14) stoploss
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for stoploss
    dh = df_1d['high'].values
    dl = df_1d['low'].values
    dc = df_1d['close'].values
    tr1 = np.abs(np.diff(dc, prepend=dc[0]))
    tr2 = np.abs(dh - np.roll(dc, 1))
    tr3 = np.abs(dl - np.roll(dc, 1))
    tr2[0] = np.abs(dh[0] - dc[0])
    tr3[0] = np.abs(dl[0] - dc[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.zeros(len(df_1d))
    for i in range(14, len(tr)):
        atr_1d[i] = np.mean(tr[i-13:i+1])
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period volume MA for volume spike detection (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is datetime64[ms], index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_4h, ATR_1d, and volume MA to propagate
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_4h = ema_50_4h_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend (price > 4h EMA50) AND volume spike
            long_condition = (curr_close > h3) and (curr_close > ema50_4h) and volume_spike
            # Short: price breaks below L3 AND downtrend (price < 4h EMA50) AND volume spike
            short_condition = (curr_close < l3) and (curr_close < ema50_4h) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below L3 (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above H3 (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3_L3_Breakout_4hEMA50_Trend_VolumeSpike_1dATRStop_v1"
timeframe = "1h"
leverage = 1.0