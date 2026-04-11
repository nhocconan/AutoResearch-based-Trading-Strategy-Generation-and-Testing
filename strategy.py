#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 12h OHLC for Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels: H3, L3 (resistance/support)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Shift by 1 to use only completed 12h bars
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Align 12h Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.008 * price_close  # ATR > 0.8% of price
        
        # Long conditions: price breaks above 12h Camarilla H3 with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_high > camarilla_h3_aligned[i])
        
        # Short conditions: price breaks below 12h Camarilla L3 with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_low < camarilla_l3_aligned[i])
        
        # ATR-based stoploss and profit target
        if position == 1:
            # Stoploss: 2.5 * ATR below entry (tracked via position logic)
            # Profit target: exit when price returns to camarilla L3 level
            exit_long = (price_close < camarilla_l3_aligned[i])  # Mean reversion to support
        elif position == -1:
            # Stoploss: 2.5 * ATR above entry
            # Profit target: exit when price returns to camarilla H3 level
            exit_short = (price_close > camarilla_h3_aligned[i])  # Mean reversion to resistance
        else:
            exit_long = False
            exit_short = False
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Camarilla breakout with 12h Camarilla levels, volume confirmation, and volatility filter.
# Uses 12h Camarilla H3/L3 levels as key resistance/support zones.
# Enters long when 4h price breaks above 12h Camarilla H3 with volume confirmation (>1.3x average)
# and sufficient volatility (ATR > 0.8% of price). Enters short when price breaks below 12h Camarilla L3
# under same conditions. Exits when price returns to the opposite Camarilla level (mean reversion).
# Works in both bull and bear markets by capturing breakouts and mean reversion at key levels.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.
# Volume confirmation ensures institutional participation. Volatility filter prevents whipsaws.