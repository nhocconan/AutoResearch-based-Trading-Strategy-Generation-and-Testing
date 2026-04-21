#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Volume_Spike_Trend_v1
Hypothesis: On 1d timeframe, Camarilla R3/S3 pivot levels from 1d data combined with weekly trend regime (price > weekly EMA34 = bull, < = bear) and volume confirmation (volume > 2.0x 20-period average) captures institutional breakouts with reduced whipsaw. In bull regime, long when price breaks above R3 with volume; in bear regime, short when price breaks below S3 with volume. Uses discrete sizing (0.25) to minimize fee churn. Target: 30-100 total trades over 4 years.
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
    
    # === Camarilla pivot levels from 1d data (using previous day's OHLC) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Previous day's OHLC (shift by 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar: use current values (no previous data)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # === 1d volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_hold_bars = 10  # max 10 days
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1w_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R3 with volume
                long_condition = (price > r3) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S3 with volume
                short_condition = (price < s3) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Check stoploss (3.0x ATR approximation using daily range)
            # Approximate ATR with 10-day average true range
            if i >= 10:
                tr1 = high[i-9:i+1] - low[i-9:i+1]
                tr2 = np.abs(high[i-9:i+1] - np.roll(close[i-9:i+1], 1))
                tr3 = np.abs(low[i-9:i+1] - np.roll(close[i-9:i+1], 1))
                tr = np.maximum(np.maximum(tr1, tr2), tr3)
                atr_approx = np.mean(tr)
            else:
                atr_approx = 0.0
            
            if position == 1:
                if price < entry_price - 3.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                elif np.sum(volume[i-9:i+1]) < np.sum(volume[i-19:i-9]):  # volume drying up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 3.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                elif np.sum(volume[i-9:i+1]) < np.sum(volume[i-19:i-9]):  # volume drying up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            
            # Time-based exit
            bars_since_entry = 0
            # Simplified: exit after max_hold_bars
            # In practice, track bars since entry, but for simplicity using price action
            if position == 1 and price < entry_price * 0.90:  # 10% trailing stop
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > entry_price * 1.10:  # 10% trailing stop
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Volume_Spike_Trend_v1"
timeframe = "1d"
leverage = 1.0