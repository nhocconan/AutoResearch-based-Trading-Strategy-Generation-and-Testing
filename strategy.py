#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum. 
1d EMA34 filter ensures alignment with daily trend. 
Volume spike (>2x 20-bar MA) confirms institutional participation.
ATR-based stoploss manages risk. Works in bull markets via upside breakouts 
and bear markets via downside breakdowns. Discrete sizing (0.30) balances 
profit potential with fee drag control. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need 34 for EMA + 1 buffer
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if np.isnan(tr[i-13:i+1]).any():
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Donchian(20) on primary 12h timeframe
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need enough for Donchian, EMA34, volume MA, and ATR
    start_idx = max(35, 20, 14)  # 35 for EMA34, 20 for Donchian/vol, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        donch_upper = upper[i]
        donch_lower = lower[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Trend filter: price relative to 1d EMA34
        price_above_ema = curr_close > ema_34_val
        price_below_ema = curr_close < ema_34_val
        
        # Breakout conditions
        breakout_upper = curr_close > donch_upper
        breakout_lower = curr_close < donch_lower
        
        if position == 0:
            # Long: break above upper Donchian + price above 1d EMA34 + volume confirmation
            long_signal = breakout_upper and price_above_ema and volume_confirm
            # Short: break below lower Donchian + price below 1d EMA34 + volume confirmation
            short_signal = breakout_lower and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_at_entry = atr_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_at_entry = atr_val
        elif position == 1:
            # Long position management
            # Stoploss: 2.5 * ATR below entry
            stop_price = entry_price - 2.5 * atr_at_entry
            # Exit conditions: stop hit OR price closes below lower Donchian
            if curr_low <= stop_price or curr_close < donch_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Stoploss: 2.5 * ATR above entry
            stop_price = entry_price + 2.5 * atr_at_entry
            # Exit conditions: stop hit OR price closes above upper Donchian
            if curr_high >= stop_price or curr_close > donch_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0