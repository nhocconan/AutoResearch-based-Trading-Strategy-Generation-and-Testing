#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 2.0 * 12h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 2.0 * 12h volume MA(20).
- Exit: ATR-based stoploss (2.5 * ATR(14)) and time-based exit (hold max 5 bars).
- Signal size: 0.25 discrete to control fee drag.
- Designed to capture strong trending moves with volume confirmation while avoiding choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian(20), volume MA(20), and ATR(14)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian(20) for 12h timeframe
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 12h timeframe
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 12h timeframe
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14, 20)  # EMA34 needs 34, Donchian needs 20, ATR needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_since_entry = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_12h[i]
        
        if position == 0:
            bars_since_entry = 0
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above Donchian(20) high AND price > 1d EMA34 (uptrend)
                if curr_high > donch_high[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: price breaks below Donchian(20) low AND price < 1d EMA34 (downtrend)
                elif curr_low < donch_low[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions
            stoploss_hit = False
            take_profit = False
            time_exit = bars_since_entry >= 5  # max 5 bars hold
            
            if position == 1:  # Long position
                # Stoploss: 2.5 * ATR below entry
                stoploss = entry_price - 2.5 * curr_atr
                stoploss_hit = curr_low <= stoploss
                # Time-based exit
                if stoploss_hit or time_exit:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Stoploss: 2.5 * ATR above entry
                stoploss = entry_price + 2.5 * curr_atr
                stoploss_hit = curr_high >= stoploss
                # Time-based exit
                if stoploss_hit or time_exit:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0