#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v3
Hypothesis: 4h Camarilla R1/S1 breakouts in direction of 1d EMA34 trend with volume confirmation (>2.0x 20-bar MA). 
In bull daily regime (price > daily EMA34), take longs on R1 breakouts; in bear daily regime (price < daily EMA34), take shorts on S1 breakouts. 
Daily EMA34 provides stable trend filter; Camarilla levels provide intraday structure; volume filter ensures participation. 
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year) by requiring confluence of breakout, daily trend, and volume.
ATR-based stoploss (2.0x) and discrete sizing (0.25) control risk and minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 4h Camarilla levels (based on previous day's OHLC) ===
    # For each 4h bar, we use the previous 1d bar's OHLC to calculate Camarilla levels
    # Since we're on 4h timeframe, we need to align the daily OHLC to each 4h bar
    
    # Get daily OHLC arrays
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    range_1d = high_1d - low_1d
    camarilla_r1_1d = close_1d + range_1d * 1.1 / 12
    camarilla_s1_1d = close_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (each 4h bar gets the previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > ema_34_1d_val
        is_bear = price < ema_34_1d_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long R1 breakouts
                long_condition = (price > r1_val) and vol_conf
                short_condition = False  # No shorts in bull regime
            else:  # bear regime
                # Bear regime: short S1 breakdowns
                short_condition = (price < s1_val) and vol_conf
                long_condition = False  # No longs in bear regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0