#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Designed for low trade frequency (~12-25/year) to minimize fee drag and improve generalization across bull/bear markets.
Uses 12h primary timeframe with 1d HTF for trend and volume context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Camarilla pivot levels (R1, S1) on 12h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla levels using previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        camarilla_r1 = r1[i]
        camarilla_s1 = s1[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume spike > 2.0 + price above 1d EMA34
            if price_close > camarilla_r1 and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below Camarilla S1 + volume spike > 2.0 + price below 1d EMA34
            elif price_close < camarilla_s1 and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit logic: reverse signal or loss of momentum
            if position == 1:
                # Exit long: price drops below Camarilla S1 OR loss of 1d trend
                if price_close < camarilla_s1 or price_close < trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price rises above Camarilla R1 OR loss of 1d trend
                if price_close > camarilla_r1 or price_close > trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0