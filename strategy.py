#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d ATR-based stoploss
# - Long when close breaks above Donchian(20) high AND 12h volume > 1.5x 20-period average
# - Short when close breaks below Donchian(20) low AND 12h volume > 1.5x 20-period average
# - Exit when price retraces to Donchian(20) midpoint OR ATR stoploss triggered
# - Uses discrete position sizing 0.25 to minimize fee churn
# - Target: 25-40 trades/year on 4h (100-160 total over 4 years)
# - Works in bull/bear: volume confirms breakout authenticity, ATR stop manages risk in volatile markets

name = "4h_12h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = np.full_like(close_4h, np.nan, dtype=float)
    donchian_low = np.full_like(close_4h, np.nan, dtype=float)
    donchian_mid = np.full_like(close_4h, np.nan, dtype=float)
    
    for i in range(19, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Pre-compute 12h volume average (20-period)
    volume_12h = df_12h['volume'].values
    vol_ma_12h = np.full_like(volume_12h, np.nan, dtype=float)
    
    for i in range(19, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Pre-compute 4h ATR(14) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        # Initial ATR (simple average)
        atr[13] = np.nanmean(tr[1:14])
        # Wilder smoothing
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Align HTF indicators to 4h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, prices, atr)  # 4h data already in prices
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(50, n):  # Start after warmup
        close_price = close_4h[i]
        
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_series = prices['volume'].values
        vol_spike = vol_series[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: close breaks above Donchian high AND volume spike
            if close_price > donchian_high[i] and vol_spike:
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short conditions: close breaks below Donchian low AND volume spike
            elif close_price < donchian_low[i] and vol_spike:
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price retrace to midpoint OR ATR stoploss
            exit_long = False
            exit_short = False
            
            if position == 1:  # Long position
                # Exit if price retraces to midpoint
                if close_price <= donchian_mid[i]:
                    exit_long = True
                # ATR stoploss: exit if price drops below entry - ATR*multiplier
                elif close_price < entry_price - atr_stop_multiplier * atr_aligned[i]:
                    exit_long = True
            else:  # Short position
                # Exit if price retraces to midpoint
                if close_price >= donchian_mid[i]:
                    exit_short = True
                # ATR stoploss: exit if price rises above entry + ATR*multiplier
                elif close_price > entry_price + atr_stop_multiplier * atr_aligned[i]:
                    exit_short = True
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals