#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dFilter_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h trend filter (EMA50) and 1d volume regime filter.
4h EMA50 determines trend: price above = bullish bias (long only), below = bearish bias (short only).
1h entry requires break of R1/S1 with volume spike confirmation. 1d ATR regime filter avoids low-volatility chop.
Designed for 15-30 trades/year to stay within fee limits while capturing directional moves in bull/bear markets.
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
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.0 / 12.0
    S1 = PP - (high_1d - low_1d) * 1.0 / 12.0
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1h volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # 1d ATR(14) for volatility regime filter - avoid low volatility chop
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(50, 20, 14, 10)  # EMA, volume avg, ATR, ATR MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_regime = atr_1d_aligned[i]
        ema_trend = ema_4h_aligned[i]
        size = 0.20  # 20% position size to manage risk and reduce fee churn
        
        if position == 0:
            # Flat - look for breakout in direction of 4h trend with volume confirmation
            # Long: price above 4h EMA50 AND break above R1 + volume spike + adequate volatility
            long_entry = (close_val > ema_trend) and (close_val > R1_aligned[i]) and \
                         volume_spike[i] and (vol_regime > 0)  # vol_regime > 0 ensures data validity
            # Short: price below 4h EMA50 AND break below S1 + volume spike + adequate volatility
            short_entry = (close_val < ema_trend) and (close_val < S1_aligned[i]) and \
                          volume_spike[i] and (vol_regime > 0)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S1 retracement or volatility expansion stop
            exit_condition = (close_val < S1_aligned[i]) or \
                           (vol_regime < 0.5 * atr_1d_aligned[i-1] if i > 0 and not np.isnan(atr_1d_aligned[i-1]) else False)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or volatility expansion stop
            exit_condition = (close_val > R1_aligned[i]) or \
                           (vol_regime < 0.5 * atr_1d_aligned[i-1] if i > 0 and not np.isnan(atr_1d_aligned[i-1]) else False)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dFilter_v1"
timeframe = "1h"
leverage = 1.0