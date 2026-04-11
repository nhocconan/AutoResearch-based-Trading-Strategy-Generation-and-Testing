#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long: price breaks above H4 Camarilla level, volume > 1.8x 24-period avg, price > 1d EMA(50)
# - Short: price breaks below L4 Camarilla level, volume > 1.8x 24-period avg, price < 1d EMA(50)
# - Exit: price returns to Camarilla H3/L3 levels or opposite H4/L4 break
# - Uses 1d EMA(50) for trend bias and 1w EMA(200) for major trend filter
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide institutional support/resistance that work in both bull and bear markets

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
    
    # Load 1w data ONCE before loop for major trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return signals
    
    # Pre-compute 1d EMA(50) for trend bias
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1w EMA(200) for major trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute 12h volume confirmation (24-period average for 12h timeframe)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_24[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Calculate Camarilla pivot levels for 12h timeframe
        # Using previous bar's high, low, close (standard Camarilla calculation)
        if i > 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            range_val = prev_high - prev_low
            
            # Camarilla levels
            h4 = pivot + (range_val * 1.1 / 2)
            h3 = pivot + (range_val * 1.1 / 4)
            l3 = pivot - (range_val * 1.1 / 4)
            l4 = pivot - (range_val * 1.1 / 2)
        else:
            # Not enough data for Camarilla calculation
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 24-period average
        vol_confirm = volume_current > 1.8 * volume_sma_24[i]
        
        # Trend filters
        # 1d EMA(50) for intermediate trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # 1w EMA(200) for major trend filter (avoid trading against major trend)
        major_trend_up = close_price > ema_200_1w_aligned[i]
        major_trend_down = close_price < ema_200_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above H4 Camarilla level with volume confirmation
        # and aligned with both intermediate and major trends
        if (close_price > h4 and vol_confirm and 
            ema_bias_long and major_trend_up):
            enter_long = True
        
        # Short breakout: price closes below L4 Camarilla level with volume confirmation
        # and aligned with both intermediate and major trends
        if (close_price < l4 and vol_confirm and 
            ema_bias_short and major_trend_down):
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to H3 level or breaks below L4 (reversal)
            exit_long = close_price < h3 or close_price < l4
        elif position == -1:
            # Exit short if price returns to L3 level or breaks above H4 (reversal)
            exit_short = close_price > l3 or close_price > h4
        
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