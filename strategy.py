#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long: price breaks above Camarilla H3 level, volume > 1.8x 24-period avg, price > 1d EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.8x 24-period avg, price < 1d EMA(50)
# - Exit: price returns to Camarilla pivot point (midpoint)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide institutional support/resistance; volume confirms breakout strength;
#   1d EMA filter ensures alignment with daily trend, reducing false breakouts in choppy markets

name = "12h_1d_camarilla_breakout_v2"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h Camarilla levels (using previous bar's OHLC)
    # Camarilla levels calculated from previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar uses current close as previous
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels for each bar based on previous bar
    range_val = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_h3 = camarilla_pivot + (range_val * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (range_val * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (range_val * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (range_val * 1.1 / 2)
    
    # Pre-compute 12h volume confirmation (24-period average - 2 days of 12h bars)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(volume_sma_24[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = camarilla_h3[i]
        l3_level = camarilla_l3[i]
        pivot_point = camarilla_pivot[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average
        vol_confirm = volume_current > 1.8 * volume_sma_24[i]
        
        # 1d EMA trend filter
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
        
        # Exit conditions: return to Camarilla pivot point
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when price returns to or below pivot point
            exit_long = close_price <= pivot_point
        elif position == -1:
            # Exit short when price returns to or above pivot point
            exit_short = close_price >= pivot_point
        
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