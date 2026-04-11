#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# - Long: price breaks above Camarilla H3 level, volume > 1.3x 20-period average, price > 1d EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.3x 20-period average, price < 1d EMA(50)
# - Exit: price returns to Camarilla Pivot point (mid-level)
# - Uses discrete position sizing (0.25) to limit fee drag
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; EMA filter adds trend bias for breakouts
# - Volume confirmation reduces false breakouts

name = "12h_1d_camarilla_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h Camarilla levels (based on previous day's OHLC)
    # We need to calculate daily OHLC from 12h data, but since we don't have direct access
    # to daily data in the loop, we'll use a rolling window approach approximation
    # For Camarilla, we typically use previous day's range
    
    # Calculate 12h ATR-like range for volatility (24-period = 2 days)
    atr_24 = pd.Series(high - low).rolling(window=24, min_periods=24).mean().values
    
    # Approximate Camarilla levels using volatility-based bands
    # Camarilla H3 = Close + 1.1 * (High - Low) * 1.1
    # Camarilla L3 = Close - 1.1 * (High - Low) * 1.1
    # Camarilla Pivot = (High + Low + Close) / 3
    
    # Use 24-period lookback to approximate daily range
    highest_24 = pd.Series(high).rolling(window=24, min_periods=24).max().values
    lowest_24 = pd.Series(low).rolling(window=24, min_periods=24).min().values
    
    # Camarilla levels based on 24-period range
    camarilla_h3 = close + 1.1 * (highest_24 - lowest_24) * 1.1 / 2
    camarilla_l3 = close - 1.1 * (highest_24 - lowest_24) * 1.1 / 2
    camarilla_pivot = (highest_24 + lowest_24 + close) / 3
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_pivot[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = camarilla_h3[i]
        l3_level = camarilla_l3[i]
        pivot_level = camarilla_pivot[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume confirmation, long bias
        if close_price > h3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume confirmation, short bias
        if close_price < l3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Camarilla Pivot
            exit_long = close_price <= pivot_level
        elif position == -1:
            # Exit short if price returns to Camarilla Pivot
            exit_short = close_price >= pivot_level
        
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