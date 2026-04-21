#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 or below S1 with 1d EMA34 trend filter and volume confirmation captures institutional breakouts. In bull trend (close > EMA34), favor longs on R1 breakouts; in bear trend (close < EMA34), favor shorts on S1 breaks. Volume > 1.5x 20-period average ensures participation. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA34 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Camarilla pivot levels (R1, S1) from previous day ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high + low + close) / 3
    
    # Calculate daily pivot points (using prior day's OHLC)
    # We need to resample to daily but using mtf_data would be incorrect here
    # Instead, we'll calculate Camarilla levels using rolling window approximation
    # For 4h data, we use the last 6 bars (~1 day) to approximate prior day's OHLC
    lookback = 6  # 6 * 4h = 24h approx
    
    # Rolling max/min/close for prior day approximation
    prior_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prior_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prior_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().shift(1).values
    
    # Pivot point
    pivot = (prior_high + prior_low + prior_close) / 3
    
    # Camarilla levels
    R1 = pivot + (1.1/12) * (prior_high - prior_low)
    S1 = pivot - (1.1/12) * (prior_high - prior_low)
    
    # === 4h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 6  # max 1 day (6 * 4h = 24h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(volume_confirmed[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(prior_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        daily_ema = ema_34_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull trend: long when price breaks above R1 with volume
                long_condition = (price > R1[i]) and vol_conf
            else:  # bear trend
                # Bear trend: short when price breaks below S1 with volume
                short_condition = (price < S1[i]) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.0x ATR approximation using price range)
            atr_approx = np.abs(high[i] - low[i])
            if position == 1:
                if price < entry_price - 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_v1"
timeframe = "4h"
leverage = 1.0