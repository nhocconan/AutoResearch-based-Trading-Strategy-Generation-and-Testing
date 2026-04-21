#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike_v1
Hypothesis: On daily timeframe, price breaking above weekly Camarilla R1 or below S1 with volume confirmation (>2.0x 20-day average) and trend alignment (price > weekly EMA50 for longs, < for shorts) captures institutional breakouts with reduced false signals. Weekly EMA50 provides multi-week trend bias to avoid counter-trend breakouts. Volume ensures conviction. Discrete sizing (0.25) and 4-bar minimum hold reduce fee churn. Target 50-120 trades over 4 years (12-30/year) within fee drag limits for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA20 trend, 1w for Camarilla/EMA50)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d EMA20 for trend alignment (HTF) ===
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === 1w EMA50 for HTF trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1w Camarilla pivot levels (R1, S1) based on PREVIOUS week's OHLC ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w_arr, 1)
    prev_high_1w[0] = prev_low_1w[0] = prev_close_1w[0] = np.nan
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1_1w = pivot_1w + (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    s1_1w = pivot_1w - (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 1d ATR (14-period) for stoploss ===
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d_arr = prices['close'].values
    
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d_arr, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d_arr, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 1d volume confirmation (volume > 2.0x 20-day average) ===
    volume_1d = prices['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume_1d > (2.0 * vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_1d[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close_1d_arr[i]
        ema_20_1d_val = ema_20_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Trend alignment: price above both 1d EMA20 and 1w EMA50 for long, below both for short
        uptrend = price > ema_20_1d_val and price > ema_50_1w_val
        downtrend = price < ema_20_1d_val and price < ema_50_1w_val
        
        if position == 0:
            # Long: price closes above weekly R1, uptrend alignment, volume confirmed
            long_condition = (price > r1_val) and uptrend and vol_conf
            # Short: price closes below weekly S1, downtrend alignment, volume confirmed
            short_condition = (price < s1_val) and downtrend and vol_conf
            
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
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below either indicator)
                elif price < ema_20_1d_val or price < ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above either indicator)
                elif price > ema_20_1d_val or price > ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0