#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when Bear Power < 0 (bulls in control) AND price > 1d EMA34 AND volume > 1.5x 20-period 1d average.
# Short when Bull Power > 0 (bears in control) AND price < 1d EMA34 AND volume > 1.5x 20-period 1d average.
# Uses discrete position size 0.25. Elder Ray measures bull/bear power via EMA, effective in ranging and trending markets.
# Volume spike confirms institutional participation. Works in both bull and bear markets by adapting to trend via EMA34.
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h EMA13 for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Bull Power and Bear Power ===
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 1d Indicators: EMA34 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA34)
    warmup = 50
    
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
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bull = bull_power[i]
        bear = bear_power[i]
        ema34_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if bears take control (Bear Power > 0) OR price crosses below EMA34
            if bear > 0 or price < ema34_val:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if bulls take control (Bull Power < 0) OR price crosses above EMA34
            if bull < 0 or price > ema34_val:
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
            # LONG: Bears weak (Bear Power < 0) AND price above EMA34 AND volume spike
            if bear < 0 and price > ema34_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bulls weak (Bull Power > 0) AND price below EMA34 AND volume spike
            elif bull > 0 and price < ema34_val and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0