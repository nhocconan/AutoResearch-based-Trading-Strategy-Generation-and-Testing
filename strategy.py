#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d volume spike and 1w trend filter
# - Enter long when price touches Camarilla L3 support AND 1d volume > 1.8x 20-period volume SMA AND 1w close > 1w EMA20
# - Enter short when price touches Camarilla H3 resistance AND 1d volume > 1.8x 20-period volume SMA AND 1w close < 1w EMA20
# - Exit: price moves to Camarilla H4 (for longs) or L4 (for shorts) or opposite pivot touch
# - Camarilla pivots provide mathematical support/resistance levels
# - Volume confirmation ensures institutional participation
# - 1w EMA20 filter avoids counter-trend trades in strong weekly trends
# - Target: 12-30 trades/year to minimize fee drag while capturing high-probability reversals

name = "12h_1d_1w_camarilla_volspike_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation and Camarilla calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute Camarilla levels for 1d data (based on previous day's OHLC)
    # Camarilla formula: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.125*(high-low)
    # L3 = close - 1.125*(high-low)
    # L4 = close - 1.5*(high-low)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_range = high_1d - low_1d
    h4 = close_1d + 1.5 * camarilla_range
    h3 = close_1d + 1.125 * camarilla_range
    l3 = close_1d - 1.125 * camarilla_range
    l4 = close_1d - 1.5 * camarilla_range
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute EMA20 for 1w close (trend filter)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 1w close aligned for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(20, n):  # Start after 20-bar warmup for volume SMA
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close vs EMA20
        uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Camarilla pivot touch signals (using 12h high/low for touch detection)
        touch_h3 = high[i] >= h3_aligned[i] and low[i] <= h3_aligned[i]  # Price touched H3 level
        touch_l3 = high[i] >= l3_aligned[i] and low[i] <= l3_aligned[i]  # Price touched L3 level
        touch_h4 = high[i] >= h4_aligned[i] and low[i] <= h4_aligned[i]  # Price touched H4 level (exit for longs)
        touch_l4 = high[i] >= l4_aligned[i] and low[i] <= l4_aligned[i]  # Price touched L4 level (exit for shorts)
        
        # Exit conditions
        exit_long = touch_h4  # Exit long when price reaches H4
        exit_short = touch_l4  # Exit short when price reaches L4
        exit_opposite_long = touch_l3  # Exit long when price touches opposite L3
        exit_opposite_short = touch_h3  # Exit short when price touches opposite H3
        
        # Trading logic
        if vol_confirm:
            # Long: L3 touch in uptrend
            if touch_l3 and uptrend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: H3 touch in downtrend
            elif touch_h3 and downtrend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and (exit_long or exit_opposite_long):
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and (exit_short or exit_opposite_short):
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals