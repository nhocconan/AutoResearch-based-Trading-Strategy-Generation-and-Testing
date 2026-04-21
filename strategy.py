#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_ChopRegime_ATRStop_v1
Hypothesis: Camarilla R1/S1 breakouts on 4h filtered by 1d EMA34 trend and choppiness regime (CHOP > 61.8 = range, < 38.2 = trend) with volume confirmation (>1.5x 20-period average) and ATR-based stoploss (2.0x). Uses discrete sizing (0.25) to minimize fee drag. Target: 75-200 total trades over 4 years for BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend regime and chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d Choppiness Index (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d_arr, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d_arr, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).sum().values  # Sum of TR
    
    # Max/min close over 14 periods
    max_close = pd.Series(close_1d_arr).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1d_arr).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: 100 * log10(sum(TR)/ (max_close - min_close)) / log10(14)
    range_1d = max_close - min_close
    chop_1d = np.where(
        (range_1d > 0) & ~np.isnan(range_1d),
        100 * np.log10(atr_1d / range_1d) / np.log10(14),
        50.0  # neutral when range is zero
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === 4h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime from 1d EMA34
        is_bull = price > ema_34_1d_val
        is_bear = price < ema_34_1d_val
        
        # Regime filter: only trade in trending markets (CHOP < 38.2) or strong breakouts in range
        is_trending = chop_val < 38.2
        is_range = chop_val > 61.8
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > r1_val) and vol_conf and (is_trending or (is_range and price > ema_34_1d_val * 1.005))
                short_condition = False  # avoid shorts in bull trend
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < s1_val) and vol_conf and (is_trending or (is_range and price < ema_34_1d_val * 0.995))
                long_condition = False  # avoid longs in bear trend
            
            # Additional filter: avoid breakouts in extreme range unless strong volume
            if is_range and not is_trending:
                # Require stronger volume confirmation in range
                vol_conf_strong = volume > (2.0 * vol_ma_20)
                long_condition = long_condition and vol_conf_strong
                short_condition = short_condition and vol_conf_strong
            
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
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout) or strong reversal
                elif price < s1_val or (is_bear and chop_val > 50 and price < ema_34_1d_val):
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
                # Exit if price breaks above R1 (failed breakdown) or strong reversal
                elif price > r1_val or (is_bull and chop_val > 50 and price > ema_34_1d_val):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_ChopRegime_ATRStop_v1"
timeframe = "4h"
leverage = 1.0