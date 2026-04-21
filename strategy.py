#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike_ATRStop_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and volume confirmation (volume > 2x 20-day average) captures sustained moves in both bull and bear markets. ATR-based stoploss (2.5x) limits drawdown. Discrete sizing (0.30) reduces churn. Target: 20-60 trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for HTF trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily close, Donchian channels (20-period) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian upper/lower (20-period lookback, excludes current bar)
    roll_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    roll_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = np.roll(roll_high, 1)  # previous bar's 20-period high
    donchian_lower = np.roll(roll_low, 1)   # previous bar's 20-period low
    donchian_upper[0] = donchian_lower[0] = np.nan  # first bar invalid
    
    # === Daily ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Daily volume confirmation (volume > 2x 20-day average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_conf = volume_confirmed[i]
        
        # Trend filter: price above weekly EMA50 for long, below for short
        uptrend = price > ema_50_1w_val
        downtrend = price < ema_50_1w_val
        
        if position == 0:
            # Long: price breaks above Donchian upper, uptrend filter, volume confirmed
            long_condition = (price > upper) and uptrend and vol_conf
            # Short: price breaks below Donchian lower, downtrend filter, volume confirmed
            short_condition = (price < lower) and downtrend and vol_conf
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 days to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.30 if position == 1 else -0.30
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price crosses below weekly EMA50)
                elif price < ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price crosses above weekly EMA50)
                elif price > ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0