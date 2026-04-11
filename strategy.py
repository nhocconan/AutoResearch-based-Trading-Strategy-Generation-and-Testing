#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Uses Camarilla levels from daily timeframe for structure
# - Long: Price breaks above R4 with volume > 1.5x 20-period 1d average AND 1w close > 1w open
# - Short: Price breaks below S4 with volume > 1.5x 20-period 1d average AND 1w close < 1w open
# - Exit: Opposite Camarilla level touch (R3 for longs, S3 for shorts) or reverse breakout
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide mathematical support/resistance levels that work in ranging markets
# - Volume confirmation filters out weak breakouts
# - Weekly trend filter ensures alignment with higher timeframe momentum

name = "6h_1d_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
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
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for Camarilla and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    # Camarilla uses: (H-L)*1.1/12, (H-L)*1.1/6, (H-L)*1.1/4, (H-L)*1.1/2
    # Added to close for resistance, subtracted from close for support
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_hl = prev_high - prev_low
    camarilla_multiplier = range_hl * 1.1 / 12
    
    # Resistance levels
    r1 = prev_close + camarilla_multiplier * 1
    r2 = prev_close + camarilla_multiplier * 2
    r3 = prev_close + camarilla_multiplier * 3
    r4 = prev_close + camarilla_multiplier * 4
    r5 = prev_close + camarilla_multiplier * 5  # Sometimes used
    r6 = prev_close + camarilla_multiplier * 6  # Sometimes used
    
    # Support levels
    s1 = prev_close - camarilla_multiplier * 1
    s2 = prev_close - camarilla_multiplier * 2
    s3 = prev_close - camarilla_multiplier * 3
    s4 = prev_close - camarilla_multiplier * 4
    s5 = prev_close - camarilla_multiplier * 5  # Sometimes used
    s6 = prev_close - camarilla_multiplier * 6  # Sometimes used
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w trend filter (bullish if close > open, bearish if close < open)
    weekly_bullish = (df_1w['close'] > df_1w['open']).values
    weekly_bearish = (df_1w['close'] < df_1w['open']).values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period 1d average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R4 with volume confirmation AND weekly bullish
        if (high_price > r4_aligned[i] or close_price > r4_aligned[i]) and vol_confirm and weekly_bullish_aligned[i]:
            enter_long = True
        
        # Short breakout: price breaks below S4 with volume confirmation AND weekly bearish
        if (low_price < s4_aligned[i] or close_price < s4_aligned[i]) and vol_confirm and weekly_bearish_aligned[i]:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches R3 (take profit) or breaks below S4 (stop/reverse)
            exit_long = (low_price <= r3_aligned[i]) or (low_price < s4_aligned[i])
        elif position == -1:
            # Exit short if price touches S3 (take profit) or breaks above R4 (stop/reverse)
            exit_short = (high_price >= s3_aligned[i]) or (high_price > r4_aligned[i])
        
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