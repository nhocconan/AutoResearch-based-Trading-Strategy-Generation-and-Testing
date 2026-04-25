#!/usr/bin/env python3
"""
1d Williams %R Mean Reversion + 1w EMA34 Trend + Volume Spike
Hypothesis: Williams %R(14) identifies overbought/oversold conditions on daily timeframe.
In strong uptrends (price > 1w EMA34), oversold readings (%R < -80) with volume confirmation
provide high-probability long entries. In strong downtrends (price < 1w EMA34), overbought
readings (%R > -20) with volume confirmation provide short entries.
Volume confirmation (>1.5x 20-day volume MA) filters weak signals.
ATR-based stoploss (2.5x ATR) manages risk. Designed for 1d timeframe targeting 30-80 total trades over 4 years.
Works in both bull and bear markets via weekly trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 weeks for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-day volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Williams %R(14)
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34, volume MA, ATR, and Williams %R
    start_idx = max(34, 20, 14, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        williams_r_val = williams_r[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for mean reversion signals at Williams %R extremes
            # Long: oversold (%R < -80) with volume confirmation in uptrend
            long_signal = (williams_r_val < -80) and volume_confirm and uptrend
            # Short: overbought (%R > -20) with volume confirmation in downtrend
            short_signal = (williams_r_val > -20) and volume_confirm and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2.5 * ATR below entry
            stop_loss = entry_price - 2.5 * atr_val
            # Exit conditions: Williams %R returns to neutral (> -50) OR stoploss hit OR trend turns down
            if williams_r_val > -50 or curr_close < stop_loss or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * atr_val
            # Exit conditions: Williams %R returns to neutral (< -50) OR stoploss hit OR trend turns up
            if williams_r_val < -50 or curr_close > stop_loss or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_%R_MeanReversion_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0