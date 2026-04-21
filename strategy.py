#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_1wEMA34_Trend_v1
Hypothesis: Daily Keltner Channel (EMA34 + 2.0*ATR10) breakouts with 1-week EMA34 trend filter. 
In bull trend (close > 1w EMA34), take longs on upper KC breakouts; in bear trend (close < 1w EMA34), take shorts on lower KC breakdowns. 
Volume confirmation (>1.5x 20-day average) filters low-quality breakouts. Discrete sizing (0.25) and ATR-based stoploss (1.5x) reduce churn.
Designed to capture sustained trends while avoiding whipsaws in ranging markets. Target: 50-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 50):
        return np.zeros(n)
    
    # === 1-week EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily ATR (10-period) for Keltner Channel and stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    # === Daily EMA34 (middle of Keltner Channel) ===
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Keltner Channel bands ===
    kc_upper = ema_34 + 2.0 * atr
    kc_lower = ema_34 - 2.0 * atr
    
    # === Volume confirmation (>1.5x 20-day average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_34[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_34_1w_val
        is_bear = price < ema_34_1w_val
        
        if position == 0:
            if is_bull:
                # Bull trend: long breakouts favored
                long_condition = (price > kc_upper[i]) and vol_conf
                short_condition = (price < kc_lower[i]) and vol_conf and (price < ema_34_1w_val * 0.99)  # stricter for shorts
            else:  # bear trend
                # Bear trend: short breakdowns favored
                short_condition = (price < kc_lower[i]) and vol_conf
                long_condition = (price > kc_upper[i]) and vol_conf and (price > ema_34_1w_val * 1.01)  # stricter for longs
            
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
            
            # Check stoploss (1.5x ATR)
            if position == 1:
                if price < entry_price - 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price re-enters Keltner Channel (failed breakout)
                elif price < kc_upper[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price re-enters Keltner Channel (failed breakdown)
                elif price > kc_lower[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_1wEMA34_Trend_v1"
timeframe = "1d"
leverage = 1.0