#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) is false AND EMA34 rising AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 (close < EMA13) AND Bull Power > 0 (close > EMA13) is false AND EMA34 falling AND volume > 1.5x 20-period average.
# Exit when Elder Power reverses sign or price crosses EMA13.
# Uses discrete position size 0.25. Designed to capture momentum shifts in trending markets with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Works in both bull and bear markets by requiring EMA34 direction and volume confirmation to filter false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13  # Bull Power = Close - EMA13
    bear_power = ema13 - close  # Bear Power = EMA13 - Close
    
    # === 1d Indicators: EMA34 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_rising = ema34_1d_aligned > np.roll(ema34_1d_aligned, 1)
    ema34_falling = ema34_1d_aligned < np.roll(ema34_1d_aligned, 1)
    # Handle first value
    ema34_rising[0] = False
    ema34_falling[0] = False
    
    # === Volume Spike: volume > 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 6h ATR for stoploss
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_raw = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power becomes negative OR price crosses below EMA13
            if bull_power[i] <= 0 or price < ema13[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power becomes negative OR price crosses above EMA13
            if bear_power[i] <= 0 or price > ema13[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power <= 0 AND EMA34 rising AND volume spike
            if bull_power[i] > 0 and bear_power[i] <= 0 and ema34_rising[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power > 0 AND Bull Power <= 0 AND EMA34 falling AND volume spike
            elif bear_power[i] > 0 and bull_power[i] <= 0 and ema34_falling[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_EMA34Trend_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0