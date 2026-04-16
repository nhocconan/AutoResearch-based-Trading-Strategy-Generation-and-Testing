#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA50 trend filter and volume spike confirmation.
# Long when Bull Power > 0, Bear Power < 0, price > 1d EMA50, and 6h volume > 1.5x 20-period average.
# Short when Bear Power < 0, Bull Power > 0, price < 1d EMA50, and 6h volume > 1.5x 20-period average.
# Exit on opposite Elder Ray signal or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# volume confirmation and trend alignment via 1d EMA50.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray Index (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 6h Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # === 1d Indicators: EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for 1d EMA50)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or np.isnan(ema50_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        ema50 = ema50_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Elder Ray turns bearish (Bear Power > 0 AND Bull Power < 0)
            if bear_power[i] > 0 and bull_power[i] < 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Elder Ray turns bullish (Bull Power > 0 AND Bear Power < 0)
            if bull_power[i] > 0 and bear_power[i] < 0:
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
            # LONG: Bull Power > 0, Bear Power < 0, price > 1d EMA50, volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and price > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0, Bull Power > 0, price < 1d EMA50, volume spike
            elif bear_power[i] < 0 and bull_power[i] > 0 and price < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_VolumeSpike_ATRStop_V1"
timeframe = "6h"
leverage = 1.0