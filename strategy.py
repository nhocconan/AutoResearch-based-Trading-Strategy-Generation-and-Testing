#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR stoploss
# - Long when price breaks above Donchian(20) high AND 1w volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND 1w volume > 1.5x 20-period average
# - Exit when price crosses Donchian(10) midpoint OR ATR-based stoploss hit (close-based)
# - Uses discrete position sizing 0.25 to minimize fee churn
# - Target: 20-40 trades/year on 1d (80-160 total over 4 years)
# - Works in bull/bear: volume confirms participation, ATR stoploss manages risk,
#   Donchian breakouts capture trends in both directions

name = "1d_donchian_20_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) for breakout signals
    donchian_high_20 = np.full_like(close, np.nan, dtype=float)
    donchian_low_20 = np.full_like(close, np.nan, dtype=float)
    for i in range(19, len(high)):
        donchian_high_20[i] = np.max(high[i-19:i+1])
        donchian_low_20[i] = np.min(low[i-19:i+1])
    
    # Donchian(10) for exit signals
    donchian_high_10 = np.full_like(close, np.nan, dtype=float)
    donchian_low_10 = np.full_like(close, np.nan, dtype=float)
    for i in range(9, len(high)):
        donchian_high_10[i] = np.max(high[i-9:i+1])
        donchian_low_10[i] = np.min(low[i-9:i+1])
    
    # Donchian(20) midpoint for alternative exit
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Pre-compute 1d ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        # Initial ATR (simple average)
        atr[13] = np.nanmean(tr[1:14])
        # Wilder smoothing
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1w volume average (20-period)
    volume_1w = df_1w['volume'].values
    vol_ma_1w = np.full_like(volume_1w, np.nan, dtype=float)
    for i in range(19, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align HTF indicators to 1d timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_20[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_1w = df_1w['volume'].values
        vol_spike = not np.isnan(vol_ma_1w_aligned[i]) and vol_1w[i] > 1.5 * vol_ma_1w_aligned[i]
        
        close_price = close[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian(20) high AND volume spike
            if close_price > donchian_high_20[i] and vol_spike:
                position = 1
                entry_price = close_price
                highest_since_entry = close_price
                lowest_since_entry = close_price
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian(20) low AND volume spike
            elif close_price < donchian_low_20[i] and vol_spike:
                position = -1
                entry_price = close_price
                highest_since_entry = close_price
                lowest_since_entry = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, close_price)
            else:
                lowest_since_entry = min(lowest_since_entry, close_price)
            
            # Exit conditions:
            # 1. Price crosses Donchian(10) midpoint (opposite direction)
            exit_mid = (position == 1 and close_price < donchian_high_10[i]) or \
                       (position == -1 and close_price > donchian_low_10[i])
            
            # 2. ATR-based stoploss (2.5 * ATR from extreme)
            atr_stop = False
            if position == 1:
                atr_stop = close_price < highest_since_entry - 2.5 * atr[i]
            else:
                atr_stop = close_price > lowest_since_entry + 2.5 * atr[i]
            
            # 3. Donchian(20) midpoint exit (mean reversion)
            exit_mid20 = (position == 1 and close_price < donchian_mid_20[i]) or \
                         (position == -1 and close_price > donchian_mid_20[i])
            
            if exit_mid or atr_stop or exit_mid20:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals