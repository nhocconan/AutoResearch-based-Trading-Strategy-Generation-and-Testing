#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakouts aligned with daily EMA34 trend regime and volume confirmation (>2x 20-period average) capture institutional moves with reduced whipsaw. 
In bull regime (daily close > daily EMA34), favor longs on R3 breakout; in bear regime (daily close < daily EMA34), favor shorts on S3 breakout. 
Volume confirmation filters low-participation false breakouts. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Camarilla pivot levels (R3, S3) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot point (PP)
    pp = (high + low + close) / 3.0
    
    # Calculate R3 and S3 levels
    r3 = pp + 1.1 * (high - low) * 1.1 / 2.0
    s3 = pp - 1.1 * (high - low) * 1.1 / 2.0
    
    # === 12h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 4 days (8 * 12h = 96h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_confirmed[i])):
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
                # Bull regime: long when price breaks above R3
                long_condition = (price > r3[i]) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S3
                short_condition = (price < s3[i]) and vol_conf
            
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
            daily_range = df_1d['high'].values - df_1d['low'].values
            atr_approx = pd.Series(daily_range).rolling(window=14, min_periods=14).mean().values
            atr_approx_aligned = align_htf_to_ltf(prices, df_1d, atr_approx)
            
            if position == 1:
                if price < entry_price - 2.5 * atr_approx_aligned[i]:
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
                if price > entry_price + 2.5 * atr_approx_aligned[i]:
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

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0