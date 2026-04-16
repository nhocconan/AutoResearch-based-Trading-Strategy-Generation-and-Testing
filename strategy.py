#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter and 1d volume confirmation.
# Long when 1h EMA(9) > EMA(21) AND 4h EMA(50) rising AND 1d volume > 1.5x 20-period average.
# Short when 1h EMA(9) < EMA(21) AND 4h EMA(50) falling AND 1d volume > 1.5x 20-period average.
# Exit on opposite EMA crossover or ATR-based stoploss (1.5*ATR from entry).
# Uses discrete position size 0.20. Session filter 08-20 UTC reduces noise.
# Target: 80-160 total trades over 4 years (20-40/year) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: EMA(9) and EMA(21) ===
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 4h Indicators: EMA(50) for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema50_rising = ema50_4h_aligned > np.roll(ema50_4h_aligned, 1)
    ema50_falling = ema50_4h_aligned < np.roll(ema50_4h_aligned, 1)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for 4h EMA50)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_1h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema9_val = ema9[i]
        ema21_val = ema21[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h[i]
        trend_up = ema50_rising[i]
        trend_down = ema50_falling[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if EMA(9) crosses below EMA(21)
            if ema9_val < ema21_val:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if EMA(9) crosses above EMA(21)
            if ema9_val > ema21_val:
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
            # LONG: EMA(9) > EMA(21) AND 4h EMA(50) rising AND volume spike
            if ema9_val > ema21_val and trend_up and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: EMA(9) < EMA(21) AND 4h EMA(50) falling AND volume spike
            elif ema9_val < ema21_val and trend_down and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA9_21_4hEMA50_1dVolumeSpike_V1"
timeframe = "1h"
leverage = 1.0