#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R(14) mean reversion with 1w HMA(21) trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1w HMA trending up AND volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND 1w HMA trending down AND volume > 1.5x 20-period average.
# Exit on opposite Williams %R level (%R > -50 for long exit, %R < -50 for short exit) or ATR-based stoploss (2*ATR).
# Uses discrete position size 0.25. Designed to capture mean reversion in extreme conditions while respecting weekly trend.
# Works in both bull and bear markets by requiring 1w trend filter (HMA direction) and volume confirmation, avoiding false mean reversion signals.
# Target: 50-120 total trades over 4 years (12-30/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 1w Indicators: HMA(21) for trend ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 10  # 21/2 = 10.5 -> 10
    sqrt_len = 4   # sqrt(21) ≈ 4.58 -> 4
    wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_1w = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    hma_up = hma_1w_aligned > np.roll(hma_1w_aligned, 1)
    hma_down = hma_1w_aligned < np.roll(hma_1w_aligned, 1)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1d ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for Williams %R/HMA/ATR)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_1d_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R rises above -50 (mean reversion complete)
            if wr > -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R falls below -50 (mean reversion complete)
            if wr < -50:
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
            # LONG: Williams %R < -80 (oversold) AND HMA trending up AND volume spike
            if wr < -80 and hma_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) AND HMA trending down AND volume spike
            elif wr > -20 and hma_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_WilliamsR14_1wHMA21_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0