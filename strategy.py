#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d: mean reversion at H3/L3 with volume confirmation
# - Long: price touches L3 with volume spike and closes above open (bullish candle)
# - Short: price touches H3 with volume spike and closes below open (bearish candle)
# - Exit: price reaches opposite H3/L3 level or midpoint (mean reversion completion)
# - Uses 1d Camarilla levels calculated from prior 1d OHLC, aligned to 12h
# - Works in both bull and bear markets by fading extremes at statistical pivot levels
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "12h_1d_camarilla_meanrev_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    # H4 = close + 1.1*(high-low)*1.1/2
    # L4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_h4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels for current day)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        open_price_i = open_price[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict filter)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Price position relative to Camarilla levels
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        
        # Bullish/bearish candle confirmation
        bullish_candle = close_price > open_price_i
        bearish_candle = close_price < open_price_i
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price touches L3 with volume and bullish candle (mean reversion long)
        if low_price <= l3 and vol_confirm and bullish_candle:
            enter_long = True
        
        # Short: price touches H3 with volume and bearish candle (mean reversion short)
        if high_price >= h3 and vol_confirm and bearish_candle:
            enter_short = True
        
        # Exit conditions: mean reversion completion
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches H3 (opposite level) or midpoint
            exit_long = close_price >= h3 or close_price >= (h3 + l3) / 2
        elif position == -1:
            # Exit short if price reaches L3 (opposite level) or midpoint
            exit_short = close_price <= l3 or close_price <= (h3 + l3) / 2
        
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