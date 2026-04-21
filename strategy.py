#!/usr/bin/env python3
"""
4h_KAMA_Trend_Regime_ChopFilter_VolumeSpike_v1
Hypothesis: KAMA adapts to market regime (trending/choppy) reducing whipsaw in bear markets. Combined with 1d trend filter, volume spike (>2x 20-bar avg), and choppiness filter (CHOP > 61.8 = choppy -> mean reversion at Camarilla S1/R1). Discrete sizing 0.25, min holding 4 bars. Target 60-120 trades over 4 years (15-30/year). Works in bull via trend-following KAMA breaks, in bear via mean reversion in chop at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for HTF trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h close, KAMA (ER=10) for adaptive trend ===
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, 10))
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0) if len(close) > 1 else 0
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 4h Choppiness Index (CHOP) regime filter ===
    # CHOP > 61.8 = choppy (mean revert), CHOP < 38.2 = trending
    atr_14 = tr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (highest_high - lowest_low)) / np.log10(14)
    chop_regime = chop > 61.8  # True = choppy, False = trending
    
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
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(chop_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        kama_val = kama[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        choppy = chop_regime[i]
        
        if position == 0:
            # In choppy regime: mean reversion at Camarilla levels
            # Long: price closes below S1 (oversold) in chop
            # Short: price closes above R1 (overbought) in chop
            long_condition = choppy and (price < s1_val) and vol_conf
            short_condition = choppy and (price > r1_val) and vol_conf
            
            # In trending regime: follow KAMA break with 1d trend filter
            # Long: price > KAMA and price > 1d EMA50 (uptrend)
            # Short: price < KAMA and price < 1d EMA50 (downtrend)
            uptrend = price > ema_50_1d_val
            downtrend = price < ema_50_1d_val
            long_condition |= (not choppy) and (price > kama_val) and uptrend and vol_conf
            short_condition |= (not choppy) and (price < kama_val) and downtrend and vol_conf
            
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
                # Exit conditions based on regime
                elif choppy and (price > pivot[i]):  # exit mean reversion at pivot
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                elif not choppy and (price < kama_val or price < ema_50_1d_val):  # exit trend break
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
                # Exit conditions based on regime
                elif choppy and (price < pivot[i]):  # exit mean reversion at pivot
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                elif not choppy and (price > kama_val or price > ema_50_1d_val):  # exit trend break
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Regime_ChopFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0