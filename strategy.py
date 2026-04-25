#!/usr/bin/env python3
"""
4h Camarilla Pivot H3L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts with volume confirmation capture momentum. 1d EMA34 ensures alignment with higher timeframe trend. Designed for 4h timeframe with 75-200 total trades over 4 years, working in both bull and bear markets via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 days for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day (1d)
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous day's OHLC (1d data)
        # Since we're on 4h timeframe, we need to get the prior day's values
        # We'll use the 1d data shifted by 1 to avoid look-ahead
        pass  # Will calculate in loop using aligned 1d data
    
    # Calculate ATR(14) for stoploss (4h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34, volume MA, ATR
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
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
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Calculate Camarilla H3/L3 from previous 1d bar (aligned)
        # Get index of completed 1d bar for current 4h bar
        # We need to use 1d data from previous day to avoid look-ahead
        h3_val = np.nan
        l3_val = np.nan
        
        # Simple approximation: use rolling window on 4h to estimate daily H/L/C
        if i >= 6:  # At least 6*4h = 24h lookback
            # Approximate daily high/low/close from last 6 4h bars
            day_high = np.max(high[i-5:i+1])
            day_low = np.min(low[i-5:i+1])
            day_close = close[i]
            
            # Camarilla levels
            range_val = day_high - day_low
            if range_val > 0:
                h3_val = day_close + range_val * 1.1 / 4
                l3_val = day_close - range_val * 1.1 / 4
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (not np.isnan(h3_val)) and (curr_close > h3_val) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (not np.isnan(l3_val)) and (curr_close < l3_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit conditions: price closes below Camarilla L3 OR stoploss hit OR EMA34 trend turns down
            if (not np.isnan(l3_val)) and curr_close < l3_val:
                signals[i] = 0.0
                position = 0
            elif curr_close < stop_loss or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above Camarilla H3 OR stoploss hit OR EMA34 trend turns up
            if (not np.isnan(h3_val)) and curr_close > h3_val:
                signals[i] = 0.0
                position = 0
            elif curr_close > stop_loss or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0