#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter and 1d volume confirmation.
# Long when 1h EMA(9) crosses above EMA(21) AND 4h close > EMA(50) AND 1d volume > 1.5x 20-day average.
# Short when 1h EMA(9) crosses below EMA(21) AND 4h close < EMA(50) AND 1d volume > 1.5x 20-day average.
# Exit on opposite EMA crossover or ATR-based stoploss (1.5*ATR from entry).
# Uses discrete position size 0.20. Designed to capture momentum with multi-timeframe confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: EMA(9) and EMA(21) ===
    close_s = pd.Series(close)
    ema9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 4h Indicators: EMA(50) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-day average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if EMA(9) crosses below EMA(21)
            if ema9[i] < ema21[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if EMA(9) crosses above EMA(21)
            if ema9[i] > ema21[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: EMA(9) crosses above EMA(21) AND 4h trend up AND volume spike
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and close[i] > ema50_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: EMA(9) crosses below EMA(21) AND 4h trend down AND volume spike
            elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and close[i] < ema50_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA9_21_4hTrend_1dVolumeSpike_V1"
timeframe = "1h"
leverage = 1.0