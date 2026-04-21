#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_HTF_VolumeRegime_Adaptive_v3
Hypothesis: Camarilla R1/S1 breakouts with adaptive volume confirmation based on 1d volatility regime (choppy vs trending).
In high volatility (trending) regime: use 1.5x volume avg for confirmation (more sensitive to breakouts).
In low volatility (choppy) regime: require 2.5x volume avg (stronger filter against false breakouts).
Volume spike confirms genuine participation. Discrete sizing (0.25) targets 20-50 trades/year.
Adapts to market conditions: more aggressive in trends, conservative in chop, improving robustness across BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d ATR (14-period) for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_14_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR percentile (20-period) to define volatility regime
    atr_percentile = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).quantile(0.5).values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Volatility regime: high vol if ATR > median (trending), low vol if ATR <= median (choppy)
    high_vol_regime = atr_1d_aligned > atr_percentile_aligned
    low_vol_regime = ~high_vol_regime
    
    # === 4h close, EMA20 for dynamic support/resistance ===
    close = prices['close'].values
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (adaptive threshold based on 1d vol regime) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    # High vol regime: 1.5x avg (more sensitive), Low vol regime: 2.5x avg (stricter)
    vol_threshold = np.where(high_vol_regime, 1.5, 2.5)
    volume_confirmed = volume > (vol_threshold * vol_ma_20)
    
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
        if (np.isnan(ema_20_4h[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_20_4h_val = ema_20_4h[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        if position == 0:
            # Adaptive entry conditions based on volatility regime
            if high_vol_regime[i]:
                # High volatility (trending): more sensitive to breakouts
                long_condition = (price > r1_val) and vol_conf and (price > ema_20_4h_val)
                short_condition = (price < s1_val) and vol_conf and (price < ema_20_4h_val)
            else:  # low_vol_regime[i]
                # Low volatility (choppy): require stronger confirmation
                long_condition = (price > r1_val) and vol_conf and (price > ema_20_4h_val * 1.005)
                short_condition = (price < s1_val) and vol_conf and (price < ema_20_4h_val * 0.995)
            
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
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout) or strong adverse move
                elif price < s1_val or price < ema_20_4h_val * 0.99:
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
                # Exit if price breaks above R1 (failed breakdown) or strong adverse move
                elif price > r1_val or price > ema_20_4h_val * 1.01:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_HTF_VolumeRegime_Adaptive_v3"
timeframe = "4h"
leverage = 1.0