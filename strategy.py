#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 level with 1d EMA34 trend confirmation and volume spike (>2x 20-period average) captures strong bullish momentum. Conversely, price breaking below Camarilla S1 level with bearish 1d EMA34 trend and volume spike captures strong bearish momentum. This strategy focuses on clear breakouts with institutional participation (volume) and trend alignment to minimize whipsaw. Discrete sizing (0.25) reduces fee churn. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA34 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Camarilla pivot levels (based on previous day) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use previous day's OHLC (from 1d data)
    camarilla_R1 = np.zeros(n)
    camarilla_S1 = np.zeros(n)
    
    # Map 1d close to each 4h bar for calculation
    # We need the previous completed 1d bar's OHLC
    # align_htf_to_ltf with additional_delay_bars=1 ensures we use completed 1d bar
    prev_close_1d = align_htf_to_ltf(prices, df_1d, close_1d, additional_delay_bars=1)
    # For high/low, we need to calculate from 1d data similarly
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    prev_high_1d = align_htf_to_ltf(prices, df_1d, high_1d, additional_delay_bars=1)
    prev_low_1d = align_htf_to_ltf(prices, df_1d, low_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels using previous 1d bar's OHLC
    camarilla_R1 = prev_close_1d + (1.1 * (prev_high_1d - prev_low_1d) / 12)
    camarilla_S1 = prev_close_1d - (1.1 * (prev_high_1d - prev_low_1d) / 12)
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1[i]) or 
            np.isnan(camarilla_S1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        daily_ema = ema_34_1d_aligned[i]
        r1_level = camarilla_R1[i]
        s1_level = camarilla_S1[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R1 with volume
                long_condition = (price > r1_level) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S1 with volume
                short_condition = (price < s1_level) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of volume confirmation
            if position == 1:
                # Exit long: price breaks below S1 or volume confirmation lost
                if (price < s1_level) or (not vol_conf):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price breaks above R1 or volume confirmation lost
                if (price > r1_level) or (not vol_conf):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0