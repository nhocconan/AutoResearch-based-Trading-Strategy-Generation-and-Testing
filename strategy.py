#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v1
Hypothesis: On 6h timeframe, trade Camarilla R1/S1 breakouts only when aligned with 1d EMA50 trend and confirmed by volume spike (>2x 20-period average). In strong 1d uptrend (price > EMA50), favor longs; in downtrend (price < EMA50), favor shorts. Uses discrete sizing (0.25) and ATR-based stop (2.0x) to manage risk. Target: 80-160 trades over 4 years (20-40/year). Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend and volume regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for HTF trend ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d volume spike: volume > 2.0x 20-period average ===
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === 6h price data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h EMA20 for short-term trend alignment ===
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 2.0x 20-period average) ===
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20_6h)
    
    # === 6h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
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
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_6h[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        ema_20_6h_val = ema_20_6h[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # boolean from aligned float
        
        # Trend alignment: 1d EMA50 defines regime, 6h EMA20 for short-term filter
        uptrend_regime = price > ema_50_1d_val
        downtrend_regime = price < ema_50_1d_val
        uptrend_short = price > ema_20_6h_val
        downtrend_short = price < ema_20_6h_val
        
        if position == 0:
            # Long conditions: 1d uptrend + price breaks R1 + volume spike + short-term alignment
            long_condition = (price > r1_val) and uptrend_regime and vol_spike and uptrend_short
            # Short conditions: 1d downtrend + price breaks S1 + volume spike + short-term alignment
            short_condition = (price < s1_val) and downtrend_regime and vol_spike and downtrend_short
            
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
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout) or trend deteriorates
                elif price < s1_val or price < ema_20_6h_val:
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
                # Exit if price breaks above R1 (failed breakdown) or trend deteriorates
                elif price > r1_val or price > ema_20_6h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v1"
timeframe = "6h"
leverage = 1.0