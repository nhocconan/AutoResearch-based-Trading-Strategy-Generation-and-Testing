#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d EMA trend filter
# - Long: price breaks above Camarilla H3 level, volume > 1.3x 20-period avg, price > 1d EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.3x 20-period avg, price < 1d EMA(50)
# - Exit: price returns to Camarilla Pivot point or opposite H3/L3 level
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; breakouts capture new trends with volume confirmation

name = "12h_1d_camarilla_breakout_v1"
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
    
    # Pre-compute 12h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels calculated from daily OHLC, applied to 12h timeframe
    # We'll use 1d OHLC to calculate Camarilla levels for the current 12h bar
    # Since we're on 12h timeframe, we need to align daily data properly
    
    # Calculate Camarilla levels from 1d data
    # Camarilla formulas:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.75 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    # L1 = Close - 0.5 * (High - Low)
    # L2 = Close - 0.75 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    # We need to shift 1d data by 1 to avoid look-ahead (use previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data (shifted by 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN since we don't have previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    camarilla_h3 = close_1d_prev + 1.0 * (high_1d_prev - low_1d_prev)
    camarilla_l3 = close_1d_prev - 1.0 * (high_1d_prev - low_1d_prev)
    camarilla_h4 = close_1d_prev + 1.5 * (high_1d_prev - low_1d_prev)
    camarilla_l4 = close_1d_prev - 1.5 * (high_1d_prev - low_1d_prev)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        pivot_level = camarilla_pivot_aligned[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 1d EMA trend filter
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above H3 with volume confirmation and long bias
        if close_price > h3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price breaks below L3 with volume confirmation and short bias
        if close_price < l3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot or breaks below L3 (reversal)
            exit_long = close_price <= pivot_level or close_price < l3_level
        elif position == -1:
            # Exit short if price returns to pivot or breaks above H3 (reversal)
            exit_short = close_price >= pivot_level or close_price > h3_level
        
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