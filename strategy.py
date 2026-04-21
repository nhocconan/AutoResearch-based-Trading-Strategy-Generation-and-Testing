#!/usr/bin/env python3
"""
4h_TRIX_ZeroLine_12hTrend_Regime_VolumeSpike_v1
Hypothesis: 4h TRIX zero-line crosses with 12h trend regime (price vs 12h EMA50) and volume confirmation (>1.8x 20-bar MA). 
In bull regime (price > 12h EMA50), favor longs on TRIX crosses above zero; in bear regime (price < 12h EMA50), favor shorts on TRIX crosses below zero. 
ATR-based stoploss (2.5x) and discrete sizing (0.25) reduce churn. Target: 75-150 total trades over 4 years by requiring confluence of TRIX signal, trend, and volume.
Designed to work in bull (momentum with trend) and bear (counter-trend momentum) markets via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h EMA50 for trend regime ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 4h TRIX (15-period, signal 9) ===
    close = prices['close'].values
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1, then * 100 for percentage
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 / np.roll(ema3, 1) - 1) * 100
    # Handle first value
    trix[0] = 0
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(trix[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        trix_val = trix[i]
        trix_prev = trix[i-1]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_50_12h_val
        is_bear = price < ema_50_12h_val
        
        if position == 0:
            # TRIX zero-line cross signals
            trix_cross_up = (trix_prev <= 0) and (trix_val > 0)
            trix_cross_down = (trix_prev >= 0) and (trix_val < 0)
            
            if is_bull:
                # Bull regime: favor longs on TRIX crosses above zero
                long_condition = trix_cross_up and vol_conf
                short_condition = trix_cross_down and vol_conf and (price < ema_50_12h_val * 0.99)  # stricter for shorts
            else:  # bear regime
                # Bear regime: favor shorts on TRIX crosses below zero
                short_condition = trix_cross_down and vol_conf
                long_condition = trix_cross_up and vol_conf and (price > ema_50_12h_val * 1.01)  # stricter for longs
            
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
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if TRIX crosses below zero (momentum loss)
                elif trix_val < 0:
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
                # Exit if TRIX crosses above zero (momentum loss)
                elif trix_val > 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_TRIX_ZeroLine_12hTrend_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0