#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot levels from 4h: mean reversion at R3/S3 with volume confirmation
# - Long: price touches S3 with volume spike and closes above open (bullish rejection)
# - Short: price touches R3 with volume spike and closes below open (bearish rejection)
# - Exit: price reaches midpoint between R3 and S3 or opposite Camarilla level
# - Uses 4h Camarilla levels calculated from prior 4h OHLC, aligned to 1h
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits

name = "1h_4h_camarilla_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return signals
    
    # Pre-compute 4h Camarilla levels (based on prior 4h bar OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous 4h bar's range
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    range_4h = high_4h - low_4h
    camarilla_r4 = close_4h + 1.1 * range_4h * 1.1 / 2
    camarilla_r3 = close_4h + 1.1 * range_4h * 1.1 / 4
    camarilla_s3 = close_4h - 1.1 * range_4h * 1.1 / 4
    camarilla_s4 = close_4h - 1.1 * range_4h * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (use prior 4h bar's levels for current 4h period)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        open_price = open_[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Price position relative to Camarilla levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        midpoint = (r3 + s3) / 2
        
        # Entry conditions: mean reversion at extreme levels with volume
        enter_long = False
        enter_short = False
        
        # Long: price touches S3 with volume and bullish close (close > open)
        if low_price <= s3 and vol_confirm and close_price > open_price:
            enter_long = True
        
        # Short: price touches R3 with volume and bearish close (close < open)
        if high_price >= r3 and vol_confirm and close_price < open_price:
            enter_short = True
        
        # Exit conditions: mean reversion to midpoint or opposite extreme
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches midpoint or touches R3
            exit_long = close_price >= midpoint or high_price >= r3
        elif position == -1:
            # Exit short if price reaches midpoint or touches S3
            exit_short = close_price <= midpoint or low_price <= s3
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals