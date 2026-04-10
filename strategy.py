#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike filter and ATR trailing stop
# - Camarilla levels (H3/L3, H4/L4) from 1d provide institutional support/resistance
# - Breakout above H4 (bullish) or below L4 (bearish) with 1d volume > 1.5x 20-day average
# - Volume confirmation ensures breakout has participation, reducing false signals
# - ATR trailing stop (3.0 * ATR) locks in profits and limits drawdown
# - Position size: 0.25 to balance risk and minimize fee churn
# - Target: 20-40 trades/year on 4h (80-160 total over 4 years)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters weak moves, ATR stop manages risk

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + range_1d * 1.1/2
    camarilla_l4 = close_1d - range_1d * 1.1/2
    camarilla_h3 = close_1d + range_1d * 1.1/4
    camarilla_l3 = close_1d - range_1d * 1.1/4
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1d volume average (20-period) for spike filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute 4h ATR for trailing stop (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d volume for spike filter (aligned)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Camarilla H4 AND volume spike
            if prices['close'].iloc[i] > h4_aligned[i] and volume_spike:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < Camarilla L4 AND volume spike
            elif prices['close'].iloc[i] < l4_aligned[i] and volume_spike:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # Exit conditions: price < Camarilla L3 (profit take) OR ATR trailing stop
                exit_long = prices['close'].iloc[i] < l3_aligned[i]  # Take profit at L3
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 3.0 * atr[i]
                exit_condition = exit_long or trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Exit conditions: price > Camarilla H3 (profit take) OR ATR trailing stop
                exit_short = prices['close'].iloc[i] > h3_aligned[i]  # Take profit at H3
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 3.0 * atr[i]
                exit_condition = exit_short or trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals