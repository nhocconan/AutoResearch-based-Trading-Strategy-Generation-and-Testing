#!/usr/bin/env python3
"""
6h_1d_VWAP_Mean_Reversion_With_Volume_Regime
Hypothesis: Mean reversion to 1-day VWAP on 6b timeframe, filtered by 1-week trend and volume regime.
- Long when: price < 1d VWAP - 0.5*ATR(6h), 1w EMA50 uptrend, volume > 20-period average
- Short when: price > 1d VWAP + 0.5*ATR(6h), 1w EMA50 downtrend, volume > 20-period average
- Exit when price crosses 1d VWAP or trend reverses
VWAP acts as a dynamic fair value mean. In ranging markets, price reverts to VWAP.
In trending markets, the 1-week EMA filter ensures we only take mean-reversion trades
in the direction of the higher timeframe trend, avoiding counter-trend whipsaws.
Volume confirmation ensures participation. Targets 20-40 trades/year (80-160 over 4 years).
"""

name = "6h_1d_VWAP_Mean_Reversion_With_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for VWAP and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 1 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d VWAP: cumulative (typical price * volume) / cumulative volume ---
    typical_price = (high_6h + low_6h + close_6h) / 3.0
    pv = typical_price * volume_6h
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume_6h)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # --- 6h ATR for dynamic bands ---
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- 1w Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for ATR and VWAP stability
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Dynamic bands around VWAP
        upper_band = vwap[i] + 0.5 * atr[i]
        lower_band = vwap[i] - 0.5 * atr[i]
        
        # Determine 1w trend
        trend_up = close_6h[i] > ema50_1w_aligned[i]
        trend_down = close_6h[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for mean-reversion entries only in direction of 1w trend with volume
            if close_6h[i] < lower_band and trend_up and vol_ok:
                # Long: price below lower VWAP band + 1w uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_6h[i] > upper_band and trend_down and vol_ok:
                # Short: price above upper VWAP band + 1w downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses above VWAP OR trend turns down
                if close_6h[i] > vwap[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses below VWAP OR trend turns up
                if close_6h[i] < vwap[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals