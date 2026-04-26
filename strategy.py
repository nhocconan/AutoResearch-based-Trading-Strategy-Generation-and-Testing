#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d EMA34 trend filter and volume spike (>2x average volume). Uses discrete position sizing (0.25) to minimize fee churn. Captures strong momentum breakouts in both bull and bear markets by aligning with 1d trend and requiring extreme volume confirmation to avoid false breakouts. Camarilla levels provide reliable intraday support/resistance for breakout entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and volume calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    atr_multiplier = 2.5  # ATR stoploss multiplier
    volume_spike_multiplier = 2.0  # Volume must be 2x average
    
    # Start after warmup (need 20 for volume, 34 for EMA, 14 for ATR)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Calculate Camarilla levels for this 4h bar using previous bar's OHLC
        if i >= 1:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            rang = prev_high - prev_low
            
            # Camarilla R3 and S3 levels
            r3 = prev_close + rang * 1.1 / 4
            s3 = prev_close - rang * 1.1 / 4
        else:
            r3 = s3 = close_val
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > volume_spike_multiplier * avg_vol
        
        # Long logic: price breaks above Camarilla R3 with 1d uptrend and volume confirmation
        long_condition = (close_val > r3) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S3 with 1d downtrend and volume confirmation
        short_condition = (close_val < s3) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal (close crosses 1d EMA34)
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

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0