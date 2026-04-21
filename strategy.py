#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1
Hypothesis: On 6h timeframe, Camarilla R1/S1 breakouts filtered by 1d EMA34 trend regime and volume confirmation (volume > 1.8x 20-period average) capture institutional participation in trending markets. In bull regime (1d close > EMA34), buy R1 breakouts; in bear regime (1d close < EMA34), sell S1 breakouts. This avoids counter-trend whipsaws while maintaining sufficient trade frequency for 6h timeframe. Discrete sizing (0.25) minimizes fee churn. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Camarilla pivot levels (based on previous 6h bar) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Camarilla levels calculated from previous bar's range
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    camarilla_r4 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    
    for i in range(1, n):
        # Previous bar's OHLC
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        # Pivot and range
        pivot = (phigh + plow + pclose) / 3
        rang = phigh - plow
        
        # Camarilla levels
        camarilla_r1[i] = pclose + rang * 1.1 / 12
        camarilla_s1[i] = pclose - rang * 1.1 / 12
        camarilla_r4[i] = pclose + rang * 1.1 / 2
        camarilla_s4[i] = pclose - rang * 1.1 / 2
    
    # === 6h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 2 days (8 * 6h = 48h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        daily_ema = ema_34_1d_aligned[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R1 with volume
                long_condition = (price > r1) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S1 with volume
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
            
            # Check stoploss (2.5x ATR approximation using Camarilla width)
            # Approximate ATR as (R4-S4)/4.5 based on Camarilla formula
            approx_atr = (r4 - s4) / 4.5
            stop_distance = 2.5 * approx_atr
            
            if position == 1:
                if price < entry_price - stop_distance:
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
                if price > entry_price + stop_distance:
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

name = "6h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0