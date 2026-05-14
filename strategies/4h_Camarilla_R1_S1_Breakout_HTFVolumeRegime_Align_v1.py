#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_HTFVolumeRegime_Align_v1
Hypothesis: Use 1d volume regime (high/low volume days) to filter Camarilla breakouts on 4h. 
In high-volume 1d regimes, breakouts are more likely to sustain; in low-volume regimes, 
fade or avoid. Combines with 4h EMA20 trend filter and discrete sizing (0.25) to reduce 
trade frequency and fee drag. Target: 60-120 trades over 4 years (15-30/year). 
Works in bull/bear via volume regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for volume regime and EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d volume regime: classify as high/low volume based on 20-day median ===
    vol_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    high_vol_regime = vol_1d > vol_median_20  # raw 1d signal
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    # === 1d EMA34 for HTF trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h close, EMA20 for trend alignment ===
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
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_20_4h[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(high_vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        ema_20_4h_val = ema_20_4h[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        high_vol = high_vol_regime_aligned[i] > 0.5  # boolean from aligned float
        
        # Trend alignment: price above both indicators for long, below both for short
        uptrend = price > ema_34_1d_val and price > ema_20_4h_val
        downtrend = price < ema_34_1d_val and price < ema_20_4h_val
        
        if position == 0:
            # In high volume regimes: trade breakouts
            # In low volume regimes: require stronger alignment (both EMAs) and avoid fading
            if high_vol:
                # High volume: breakout strategy
                long_condition = (price > r1_val) and uptrend and vol_conf
                short_condition = (price < s1_val) and downtrend and vol_conf
            else:
                # Low volume: only trade with strong trend alignment, avoid false breakouts
                long_condition = (price > r1_val) and uptrend and vol_conf and (price > ema_34_1d_val * 1.005)
                short_condition = (price < s1_val) and downtrend and vol_conf and (price < ema_34_1d_val * 0.995)
            
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
                # Exit if price breaks below S1 (failed breakout) or trend deteriorates
                elif price < s1_val or price < ema_20_4h_val:
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
                # Exit if price breaks above R1 (failed breakdown) or trend deteriorates
                elif price > r1_val or price > ema_20_4h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_HTFVolumeRegime_Align_v1"
timeframe = "4h"
leverage = 1.0