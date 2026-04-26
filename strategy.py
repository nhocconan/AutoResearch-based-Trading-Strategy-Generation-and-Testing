#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d EMA34 trend filter, volume confirmation (>2x average volume), and choppiness regime filter (CHOP < 50 for trending markets). Designed for fewer trades (target 12-37/year) to minimize fee drift while capturing strong directional moves in both bull and bear markets via 1d trend alignment and regime filtering to avoid whipsaws in ranging markets.
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
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR1) / (max(high) - min(low))) / log10(N)
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # True Range itself
    sum_atr_1 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_numerator = sum_atr_1
    chop_denominator = max_high - min_low
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_ratio = chop_numerator / chop_denominator
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending markets: CHOP < 50 (middle ground for fewer whipsaws)
    chop_filter = chop < 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    # Start after warmup (need 20 for Camarilla, 34 for EMA, 14 for ATR, 14 for CHOP)
    start_idx = max(20, 34, 14, 14)
    
    for i in range(start_idx, n):
        # Need previous day's OHLC for Camarilla levels
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
            
        # Camarilla R3 and S3 levels (stronger levels)
        r3 = prev_close + range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 4
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        chop_val = chop_filter[i]
        
        # Skip if any data not ready
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume (stricter for fewer trades)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Regime filter: only trade in trending markets (CHOP < 50)
        regime_ok = chop_val
        
        # Long logic: price breaks above R3 with 1d uptrend, volume confirmation, and trending regime
        long_condition = (close_val > r3) and (close_val > ema_val) and volume_confirmed and regime_ok
        # Short logic: price breaks below S3 with 1d downtrend, volume confirmation, and trending regime
        short_condition = (close_val < s3) and (close_val < ema_val) and volume_confirmed and regime_ok
        
        # Exit logic: trend reversal (close crosses 1d EMA34) OR regime change to ranging
        exit_long = close_val < ema_val or not chop_val
        exit_short = close_val > ema_val or not chop_val
        
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

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0