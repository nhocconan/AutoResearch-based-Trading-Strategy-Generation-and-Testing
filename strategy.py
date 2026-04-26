#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA34 trend filter and volume confirmation (>1.5x average volume). Designed for lower trade frequency (target: 30-100 trades over 4 years) to minimize fee drag while capturing strong trending moves in both bull and bear markets via weekly trend alignment. Uses discrete position sizing (0.25) and ATR-based stoploss for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for HTF trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.0  # ATR stoploss multiplier
    
    # Start after warmup (need 20 for volume, 34 for weekly EMA, 14 for ATR)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Need previous day's OHLC for Camarilla levels (1d timeframe)
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous day's high, low, close (for Camarilla calculation on 1d timeframe)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels (R1 and S1 - core levels)
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position if invalid range
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R1 and S1 levels
        r1 = prev_close + range_val * 1.1 / 12
        s1 = prev_close - range_val * 1.1 / 12
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1w_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if np.isnan(r1) or np.isnan(s1) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (balanced for trade frequency)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above R1 with weekly uptrend and volume confirmation
        long_condition = (close_val > r1) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S1 with weekly downtrend and volume confirmation
        short_condition = (close_val < s1) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal (price crosses weekly EMA34)
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0