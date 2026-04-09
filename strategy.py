#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels + volume confirmation + ATR stoploss
# Camarilla pivots from 1d provide key support/resistance levels for mean reversion in ranging markets
# Long when price touches L3 level with volume confirmation, short when touches H3 level
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion effective in ranging conditions, ATR stoploss manages trend risk

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    rang = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * rang / 2
    camarilla_l3 = close_1d - 1.1 * rang / 2
    
    # Calculate 1d average volume (2-period for daily)
    vol_s_1d = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_s_1d.rolling(window=2, min_periods=2).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Pre-compute ATR(14) for stoploss on 12h timeframe
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track entry price for stoploss
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1=long, -1=short, 0=flat
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or mean reversion target
            if close[i] < entry_price[i] - 2.0 * atr[i]:  # ATR stoploss
                position = 0
                signals[i] = 0.0
            elif close[i] > camarilla_h3_aligned[i]:  # Mean reversion target reached
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or mean reversion target
            if close[i] > entry_price[i] + 2.0 * atr[i]:  # ATR stoploss
                position = 0
                signals[i] = 0.0
            elif close[i] < camarilla_l3_aligned[i]:  # Mean reversion target reached
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at Camarilla L3/H3 with volume confirmation
            if close[i] <= camarilla_l3_aligned[i] and volume_confirmed:
                position = 1
                entry_price[i] = close[i]
                signals[i] = 0.25
            elif close[i] >= camarilla_h3_aligned[i] and volume_confirmed:
                position = -1
                entry_price[i] = close[i]
                signals[i] = -0.25
    
    return signals