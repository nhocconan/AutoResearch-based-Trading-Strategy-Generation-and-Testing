# 12h_1d_camarilla_breakout_volume_v1
# Hypothesis: 12h Camarilla pivot breakout + 1d EMA trend filter + volume confirmation
# - Camarilla pivot levels (L4, L3, H3, H4) calculated from previous 1d high/low/close
# - Long when price breaks above H3 with volume > 1.5x 20-period average and price > 1d EMA200
# - Short when price breaks below L3 with volume > 1.5x 20-period average and price < 1d EMA200
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe
# - Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
# - 1d EMA200 provides strong trend filter, reducing false signals in choppy markets
# - Volume confirmation ensures breakouts have conviction

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d EMA200
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute Camarilla levels from previous 1d bar
    # Camarilla levels use previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous bar)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    h4 = close_1d + 1.5 * range_1d
    h3 = close_1d + 1.1 * range_1d
    l3 = close_1d - 1.1 * range_1d
    l4 = close_1d - 1.5 * range_1d
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    h4_prev = np.roll(h4, 1)
    h3_prev = np.roll(h3, 1)
    l3_prev = np.roll(l3, 1)
    l4_prev = np.roll(l4, 1)
    h4_prev[0] = np.nan
    h3_prev[0] = np.nan
    l3_prev[0] = np.nan
    l4_prev[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute 12h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA200 trend filter
        price_above_ema200 = price_close > ema200_1d_aligned[i]
        price_below_ema200 = price_close < ema200_1d_aligned[i]
        
        # Camarilla breakout levels
        breakout_long = price_close > h3_aligned[i]  # Break above H3
        breakout_short = price_close < l3_aligned[i]  # Break below L3
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above H3 + price above 1d EMA200 + volume confirmation
        if breakout_long and price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: Break below L3 + price below 1d EMA200 + volume confirmation
        if breakout_short and price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla break or price crosses 1d EMA200
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR price crosses below 1d EMA200
            exit_long = (price_close < l3_aligned[i]) or (not price_above_ema200)
        elif position == -1:
            # Exit short if price breaks above H3 OR price crosses above 1d EMA200
            exit_short = (price_close > h3_aligned[i]) or (not price_below_ema200)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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