#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrendRegime_v2
Hypothesis: On daily timeframe, Camarilla R1/S1 breakouts aligned with weekly EMA50 trend capture multi-week momentum with low trade frequency (7-15/year). Uses volume confirmation (1.5x 20-day average) and ATR-based stoploss (2.5x) to reduce whipsaw. Designed for BTC/ETH to work in both bull (breakouts with trend) and bear (mean-reversion at extremes) regimes. Discrete sizing 0.25 minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily ATR (20-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    # === Daily volume confirmation (volume > 1.5x 20-day average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === Daily Camarilla pivot levels (R1, S1) based on PREVIOUS day's OHLC ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first day invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_50_1w_val
        is_bear = price < ema_50_1w_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > r1_val) and vol_conf
                short_condition = (price < s1_val) and vol_conf and (price < ema_50_1w_val * 0.98)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < s1_val) and vol_conf
                long_condition = (price > r1_val) and vol_conf and (price > ema_50_1w_val * 1.02)  # stricter for longs
            
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
            
            # Minimum holding period of 5 days to reduce churn
            if bars_since_entry < 5:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout)
                elif price < s1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above R1 (failed breakdown)
                elif price > r1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrendRegime_v2"
timeframe = "1d"
leverage = 1.0