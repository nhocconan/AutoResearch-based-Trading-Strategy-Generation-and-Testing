#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 or below S1 from the prior 1d session,
combined with 1d EMA34 trend regime and volume confirmation (volume > 1.8x 20-period average),
captures institutional breakouts with reduced whipsaw. In bull 1d regime (1d close > 1d EMA34),
favor longs on R1 breakouts; in bear 1d regime (1d close < 1d EMA34), favor shorts on S1 breakouts.
Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot and EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (R1, S1) from prior day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + (1.1 * (high_1d - low_1d) / 12)
    camarilla_s1 = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Align to 4h (wait for prior 1d close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d EMA34 for trend regime ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 6  # max 1 day (6 * 4h = 24h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = prices['close'].iloc[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        regime_ema = ema_34_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # 1d trend regime
        is_bull = price > regime_ema
        is_bear = price < regime_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R1
                long_condition = (price > r1) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S1
                short_condition = (price < s1) and vol_conf
            
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
            
            # Check stoploss (2.5x ATR)
            high_price = prices['high'].iloc[i]
            low_price = prices['low'].iloc[i]
            if i >= 20:
                atr_series = pd.Series(high_price - low_price).rolling(window=14, min_periods=14).mean()
                atr_val = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 0.0
            else:
                atr_val = 0.0
            
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
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
                if price > entry_price + 2.5 * atr_val:
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0