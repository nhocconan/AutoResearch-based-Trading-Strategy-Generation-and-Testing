#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_Volume_HTFTrend
Hypothesis: 1h Camarilla pivot R1/S1 breakout with volume confirmation and 4h trend filter (EMA34).
In uptrend (price > EMA34): long R1 break + volume, short S1 breakdown only if strong.
In downtrend (price < EMA34): short S1 breakdown + volume, long R1 break only if strong.
Volume filter: >1.5x 20-period volume MA. Position size 0.20.
Targets 15-37 trades/year per symbol (60-150 total over 4 years) by using 4h for signal direction,
1h only for entry timing, and session filter (08-20 UTC) to reduce noise.
Works in both bull/bear: trend filter adapts to market regime, volume avoids false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop (4h for trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # EMA34 on 4h close
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1h Indicators (primary timeframe) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot points (R1, S1) from previous day
    # Using previous 1h bar's high, low, close (shifted by 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # 4h trend filter
        uptrend = price > ema_34_4h_aligned[i]
        downtrend = price < ema_34_4h_aligned[i]
        
        if position == 0:
            if uptrend and vol_ok:
                # In uptrend: long R1 break
                if price > r1[i]:
                    signals[i] = 0.20
                    position = 1
            elif downtrend and vol_ok:
                # In downtrend: short S1 breakdown
                if price < s1[i]:
                    signals[i] = -0.20
                    position = -1
            # Counter-trend entries only with strong volume (2.0x) and extreme price
            elif vol > 2.0 * vol_ma[i]:  # stronger volume for counter-trend
                if price < s1[i] and uptrend:  # pullback in uptrend
                    signals[i] = 0.20
                    position = 1
                elif price > r1[i] and downtrend:  # bounce in downtrend
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR loses 4h uptrend
            if price < s1[i] or price < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above R1 OR gains 4h uptrend
            if price > r1[i] or price > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_Volume_HTFTrend"
timeframe = "1h"
leverage = 1.0