#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1w EMA trend filter
# - Camarilla pivot levels (L3/L3, H3/H3) provide high-probability breakout zones
# - 1d volume spike (volume > 2x 20-day average) confirms institutional participation
# - 1w EMA trend filter ensures alignment with higher timeframe direction
# - ATR(14) trailing stop (2.5x) manages risk without whipsaw
# - Position size: 0.25 discrete to minimize fee churn
# - Target: 20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Works in bull/bear: Camarilla adapts to volatility, volume filter avoids false breakouts, trend filter avoids counter-trend trades

name = "4h_1d_1w_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # L3 = Close - Range * 1.1 / 4
    # L4 = Close - Range * 1.1 / 2
    # H3 = Close + Range * 1.1 / 4
    # H4 = Close + Range * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    l3_1d = close_1d - (range_1d * 1.1 / 4)
    l4_1d = close_1d - (range_1d * 1.1 / 2)
    h3_1d = close_1d + (range_1d * 1.1 / 4)
    h4_1d = close_1d + (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    
    # Pre-compute 1d volume spike filter: volume > 2x 20-day average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * volume_ma_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Pre-compute 1w EMA for trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
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
        if (np.isnan(l3_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or np.isnan(h4_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: 1d volume spike
        volume_confirm = volume_spike_aligned[i] > 0.5  # Boolean as float
        
        # Trend filter: price above/below 1w EMA
        trend_long = prices['close'].iloc[i] > ema_21_1w_aligned[i]
        trend_short = prices['close'].iloc[i] < ema_21_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > H3 AND volume confirmation AND uptrend
            if (prices['close'].iloc[i] > h3_1d_aligned[i] and 
                volume_confirm and trend_long):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < L3 AND volume confirmation AND downtrend
            elif (prices['close'].iloc[i] < l3_1d_aligned[i] and 
                  volume_confirm and trend_short):
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
                # Exit conditions: price < L3 (stop loss) OR ATR trailing stop
                exit_long = prices['close'].iloc[i] < l3_1d_aligned[i]  # Stop loss at L3
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]
                exit_condition = exit_long or trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Exit conditions: price > H3 (stop loss) OR ATR trailing stop
                exit_short = prices['close'].iloc[i] > h3_1d_aligned[i]  # Stop loss at H3
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]
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