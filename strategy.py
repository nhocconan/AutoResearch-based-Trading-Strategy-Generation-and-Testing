#!/usr/bin/env python3
"""
6h_VolumeSpike_Reversal_v1
Hypothesis: After extreme volume spikes (top 5% of 100-bar volume), price often reverses in 6h timeframe.
Long when volume spike + price < BB lower (20,2) + RSI < 30.
Short when volume spike + price > BB upper (20,2) + RSI > 70.
Uses 1w trend filter: only long in 1w uptrend, only short in 1w downtrend.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 50-150 total trades over 4 years via strict volume spike + BB/RSI confluence.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Bollinger Bands (20,2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume spike: top 5% of 100-bar volume
    vol_s = pd.Series(volume)
    vol_rank = vol_s.rolling(window=100, min_periods=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    volume_spike = vol_rank.values > 0.95  # Top 5%
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w EMA(34), BB(20), RSI(14), volume rank(100)
    start_idx = max(34, 20, 14, 100)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(bb_mid[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_rank[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: volume spike + price at BB lower + RSI oversold + 1w uptrend
            long_signal = vol_spike and (close_val <= bb_lower[i]) and (rsi[i] < 30) and trend_up
            
            # Short: volume spike + price at BB upper + RSI overbought + 1w downtrend
            short_signal = vol_spike and (close_val >= bb_upper[i]) and (rsi[i] > 70) and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses above BB mid OR RSI > 50 OR 1w trend flips down
            if (close_val > bb_mid[i]) or (rsi[i] > 50) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses below BB mid OR RSI < 50 OR 1w trend flips up
            if (close_val < bb_mid[i]) or (rsi[i] < 50) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_VolumeSpike_Reversal_v1"
timeframe = "6h"
leverage = 1.0