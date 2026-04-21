#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1-week EMA200 trend filter (bull/bear agnostic) and volume confirmation.
Long when price breaks above R1, above 1w EMA200, and volume > 2.0x 20-period MA.
Short when price breaks below S1, below 1w EMA200, and volume > 2.0x 20-period MA.
Uses ATR-based stop (2.0x) and minimum holding period of 4 bars.
1w EMA200 provides strong regime filter to avoid counter-trend trades in both bull and bear markets.
Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA200 trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:  # need ~4 years of weekly data for EMA200
        return np.zeros(n)
    
    # === 1w EMA200 for trend regime ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 4h Camarilla pivot levels (R1, S1) ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_200_1w_val = ema_200_1w_aligned[i]
        vol_avg = vol_ma[i]
        r1_val = r1[i]
        s1_val = s1[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirm = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1, above 1w EMA200, volume confirm
            long_condition = (price > r1_val) and (price > ema_200_1w_val) and volume_confirm
            # Short: price breaks below S1, below 1w EMA200, volume confirm
            short_condition = (price < s1_val) and (price < ema_200_1w_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 1w EMA200)
                elif price < ema_200_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 1w EMA200)
                elif price > ema_200_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0