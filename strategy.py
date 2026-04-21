#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: On 1d timeframe, Camarilla R3/S3 breakout with weekly EMA34 trend filter and volume confirmation (>1.5x 20-day average) captures institutional moves in both bull and bear markets. Weekly trend regime ensures we trade with the higher timeframe momentum, reducing whipsaw. Discrete sizing (0.25) and ATR-based stoploss (2.5x) control risk. Target: 30-80 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla levels (R3, S3) from previous day ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Camarilla levels: based on previous day's range
    R3 = np.zeros(n)
    S3 = np.zeros(n)
    for i in range(1, n):
        # Previous day's OHLC
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        range_val = phigh - plow
        if range_val > 0:
            R3[i] = pclose + range_val * 1.1 / 4
            S3[i] = pclose - range_val * 1.1 / 4
        else:
            R3[i] = pclose
            S3[i] = pclose
    
    # === Daily volume confirmation (volume > 1.5x 20-day average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 15  # max 15 days
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R3 with volume
                long_condition = (price > R3[i]) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S3 with volume
                short_condition = (price < S3[i]) and vol_conf
            
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
            
            # Check stoploss (2.5x ATR approximation using daily range)
            # Use 20-day average true range as proxy for volatility
            if i >= 20:
                atr_approx = np.mean(np.maximum(np.maximum(high[i-20:i] - low[i-20:i], 
                                                      np.abs(high[i-20:i] - np.roll(close[i-20:i], 1))), 
                                              np.abs(low[i-20:i] - np.roll(close[i-20:i], 1))))
            else:
                atr_approx = np.mean(high[max(0,i-20):i] - low[max(0,i-20):i]) if i > 0 else 1.0
            
            if position == 1:
                if price < entry_price - 2.5 * atr_approx:
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
                if price > entry_price + 2.5 * atr_approx:
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

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0