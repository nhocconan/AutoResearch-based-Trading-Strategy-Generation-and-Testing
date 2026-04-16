#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume spike confirmation.
# Long when Bull Power > 0 AND price > 1d EMA50 AND volume > 1.5x 20-period 6h average volume.
# Short when Bear Power < 0 AND price < 1d EMA50 AND volume > 1.5x 20-period 6h average volume.
# Exit when Bull/Bear Power crosses zero or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture momentum in trending markets with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by requiring alignment with 1d EMA50 trend and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 1d Indicators: EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    price_above_ema50 = close > ema50_1d_aligned
    price_below_ema50 = close < ema50_1d_aligned
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_6h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_spike = volume_spike[i]
        above_ema50 = price_above_ema50[i]
        below_ema50 = price_below_ema50[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power crosses below zero
            if bull <= 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power crosses above zero
            if bear >= 0:
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
            # LONG: Bull Power > 0 AND price > 1d EMA50 AND volume spike
            if bull > 0 and above_ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0 AND price < 1d EMA50 AND volume spike
            elif bear < 0 and below_ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0