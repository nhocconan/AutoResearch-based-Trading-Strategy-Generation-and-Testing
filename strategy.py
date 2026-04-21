#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4h1dTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h/1d EMA trend filter and volume confirmation (>1.8x 20-period MA).
Uses 4h and 1d HTF for signal direction, 1h only for entry timing precision. ATR-based stop (2.0x) and minimum holding period of 3 bars to reduce churn.
Session filter: 08-20 UTC to avoid low-liquidity hours.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Designed for 1h timeframe with dual HTF trend alignment to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h and 1d for EMA trends)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 34 or len(df_1d) < 34:
        return np.zeros(n)
    
    # === 4h EMA34 for trend regime ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.8x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1h Camarilla pivot levels (R1, S1) ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_34_4h_val = ema_34_4h_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_avg = vol_ma[i]
        r1_val = r1[i]
        s1_val = s1[i]
        
        # Volume confirmation: current volume > 1.8x average (stricter threshold)
        volume_confirm = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1, above both 4h and 1d EMA34, volume confirm
            long_condition = (price > r1_val) and (price > ema_34_4h_val) and (price > ema_34_1d_val) and volume_confirm
            # Short: price breaks below S1, below both 4h and 1d EMA34, volume confirm
            short_condition = (price < s1_val) and (price < ema_34_4h_val) and (price < ema_34_1d_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.20 if position == 1 else -0.20
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below either 4h or 1d EMA34)
                elif price < ema_34_4h_val or price < ema_34_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above either 4h or 1d EMA34)
                elif price > ema_34_4h_val or price > ema_34_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4h1dTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0