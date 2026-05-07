#!/usr/bin/env python3
# 4h_RankVWAP_Trend_Composite
# Hypothesis: Combines VWAP-based volume ranking (top 20% volume bins), EMA trend, and volatility filter to capture institutional flow days.
# VWAP bins identify high-conviction volume days; EMA50 filters trend direction; ATR filter avoids chop. Works in bull/bear by adapting to volume-driven moves.
# Target: 20-40 trades/year per symbol via strict volume-price alignment.

timeframe = "4h"
name = "4h_RankVWAP_Trend_Composite"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price for VWAP
    typical_price = (high + low + close) / 3.0
    
    # VWAP calculation
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Rank VWAP deviation: how far price is from VWAP in ATR units
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    vwap_dev = (close - vwap) / atr
    vwap_dev = np.where(atr == 0, 0, vwap_dev)
    
    # Volume ranking: percentile of volume over lookback (rank 80+ = top 20% vol days)
    vol_rank = np.zeros(n)
    lookback = 50
    for i in range(lookback, n):
        vol_window = volume[i-lookback:i]
        vol_rank[i] = (np.sum(vol_window < volume[i]) / lookback) * 100
    
    # EMA50 trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        if np.isnan(vwap_dev[i]) or np.isnan(vol_rank[i]) or np.isnan(ema50[i]) or atr[i] == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: high volume rank (top 20%), price above VWAP, uptrend
            if vol_rank[i] >= 80 and vwap_dev[i] > 0.5 and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # Short: high volume rank, price below VWAP, downtrend
            elif vol_rank[i] >= 80 and vwap_dev[i] < -0.5 and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: volume drops or price reverts to VWAP
            if vol_rank[i] < 60 or abs(vwap_dev[i]) < 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: volume drops or price reverts to VWAP
            if vol_rank[i] < 60 or abs(vwap_dev[i]) < 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals