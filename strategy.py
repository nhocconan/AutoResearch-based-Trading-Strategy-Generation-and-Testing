#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume confirmation (>1.5x average volume). Uses ATR-based stoploss and discrete position sizing (0.25) to minimize fee churn. Designed to capture strong momentum moves in both bull and bear markets by aligning with 12h trend and requiring volume confirmation to avoid false breakouts. R1/S1 levels provide tighter entry points for higher probability trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
    
    # Start after warmup (need 20 for Camarilla, 50 for EMA, 14 for ATR, 20 for volume)
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Need previous period's OHLC for Camarilla levels
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous period's high, low, close (for Camarilla calculation)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
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
            
        # Camarilla R1 and S1 levels (tighter breakout levels)
        r1 = prev_close + range_val * 1.0 / 12
        s1 = prev_close - range_val * 1.0 / 12
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_12h_aligned[i]
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
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above R1 with 12h uptrend and volume confirmation
        long_condition = (close_val > r1) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S1 with 12h downtrend and volume confirmation
        short_condition = (close_val < s1) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal (close crosses 12h EMA50)
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

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0