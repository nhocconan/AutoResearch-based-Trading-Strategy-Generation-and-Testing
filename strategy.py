#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) with 1d EMA(50) trend filter and volume confirmation.
# Williams %R measures overbought/oversold: Long when %R crosses above -80 from below (oversold bounce),
# Short when %R crosses below -20 from above (overbought rejection).
# Volume confirmation: current 12h volume > 1.5x 20-period 1d average volume (aligned).
# Trend filter: 1d EMA(50) slope > 0 for longs, < 0 for shorts (using EMA(50) - EMA(50)[1]).
# Uses discrete position size 0.25. Designed to capture mean reversions in trending markets with volume.
# Works in both bull and bear markets by requiring trend alignment and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Williams %R (14) ===
    highest_high_12h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_12h - close) / (highest_high_12h - lowest_low_12h + 1e-10)
    
    # === 1d Indicators: EMA(50) for trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: EMA(50) slope > 0 for bullish, < 0 for bearish
    ema_50_slope = ema_50_1d_aligned - np.roll(ema_50_1d_aligned, 1)
    ema_50_slope[0] = 0  # first value has no previous
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 12h ATR for stoploss
    tr1_12h = pd.Series(high).diff()
    tr2_12h = pd.Series(low).diff().abs()
    tr3_12h = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_slope[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_12h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        ema_slope = ema_50_slope[i]
        atr_val = atr_12h_raw[i]
        
        # Williams %R cross signals
        williams_r_prev = williams_r[i-1] if i > 0 else -50
        wr_cross_above_80 = williams_r_prev <= -80 and williams_r[i] > -80  # Oversold bounce
        wr_cross_below_20 = williams_r_prev >= -20 and williams_r[i] < -20   # Overbought rejection
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses below -50 (momentum loss) or ATR stop
            if williams_r[i] < -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses above -50 (momentum loss) or ATR stop
            if williams_r[i] > -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold bounce) AND EMA(50) sloping up AND volume spike
            if wr_cross_above_80 and ema_slope > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 (overbought rejection) AND EMA(50) sloping down AND volume spike
            elif wr_cross_below_20 and ema_slope < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_VolumeConfirm_V1"
timeframe = "12h"
leverage = 1.0