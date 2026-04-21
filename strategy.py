#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_HTFTrend_ATRStop_V1
Hypothesis: 1h Camarilla R1/S1 breakout with volume confirmation (>1.2x 20-period volume MA) and 4h/1d HTF trend filter (price > EMA34 on 4h AND price > EMA50 on 1d for longs; opposite for shorts). Uses ATR-based stop via signal=0 when price moves 2.0*ATR against position. Designed for low trade frequency (60-150 total over 4 years) to minimize fee drag and work in bull/bear markets via dual-HT trend confirmation and session filter (08-20 UTC).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Session filter: 08-20 UTC (precompute once) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop (4h and 1d for trend filters)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 34 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators (primary timeframe) ===
    # Need 1h high/low/close for Camarilla calculation
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Previous period's high/low/close for Camarilla levels (use prior completed bar)
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close = np.roll(close_1h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R1 and S1 levels
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # warmup for EMA34/50 and ATR
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1h[i]
        vol = volume_1h[i]
        vol_ok = vol > 1.2 * vol_ma[i]  # volume confirmation
        
        # HTF trend conditions
        uptrend_4h = price > ema_34_4h_aligned[i]
        uptrend_1d = price > ema_50_1d_aligned[i]
        downtrend_4h = price < ema_34_4h_aligned[i]
        downtrend_1d = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Camarilla R1 breakout + volume + HTF uptrend
            if price > r1[i] and vol_ok and uptrend_4h and uptrend_1d:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Camarilla S1 breakdown + volume + HTF downtrend
            elif price < s1[i] and vol_ok and downtrend_4h and downtrend_1d:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price breaks below S1 or loss of volume/uptrend
            elif price < s1[i] or not vol_ok or not (uptrend_4h and uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price breaks above R1 or loss of volume/downtrend
            elif price > r1[i] or not vol_ok or not (downtrend_4h and downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_HTFTrend_ATRStop_V1"
timeframe = "1h"
leverage = 1.0