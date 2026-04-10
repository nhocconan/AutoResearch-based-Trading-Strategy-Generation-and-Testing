#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels with 1w trend filter and volume confirmation
# - Long when price touches Camarilla L3 support AND 1w close > 1w open (bullish weekly candle) AND volume > 1.5x 20-day average volume
# - Short when price touches Camarilla H3 resistance AND 1w close < 1w open (bearish weekly candle) AND volume > 1.5x 20-day average volume
# - Exit when price moves to opposite Camarilla level (L3 to H3 or H3 to L3) or crosses the 4/5 level (midpoint)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla levels provide high-probability reversal zones in ranging markets
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Volume confirmation reduces false signals

name = "1d_1w_camarilla_pivot_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 20-day average volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla levels (based on previous day's range)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # Using previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels for each bar (based on previous day)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_mid = (camarilla_h3 + camarilla_l3) / 2  # Midpoint between H3 and L3
    
    # Pre-compute 1w trend: bullish if weekly close > weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Align HTF indicators to 1d timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_bullish_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price touches L3 support AND weekly bullish AND volume spike
            if (low[i] <= camarilla_l3[i] and 
                weekly_bullish_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price touches H3 resistance AND weekly bearish AND volume spike
            elif (high[i] >= camarilla_h3[i] and 
                  not weekly_bullish_aligned[i] and  # Weekly bearish
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price moves to opposite Camarilla level or crosses midpoint
            exit_long = (position == 1 and (high[i] >= camarilla_h3[i] or close[i] >= camarilla_mid[i]))
            exit_short = (position == -1 and (low[i] <= camarilla_l3[i] or close[i] <= camarilla_mid[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals