#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with adaptive volume threshold
# - Uses dynamic volume threshold based on volume percentile (more robust than fixed multiplier)
# - Adds momentum filter: requires price to be above/below 20-period EMA for breakout validity
# - Long: price breaks above R4 with volume in top 30% and close above EMA20
# - Short: price breaks below S4 with volume in top 30% and close below EMA20
# - Exit: mean reversion at R3/S3 levels or EMA crossover
# - Works in both bull/bear by fading extremes (R3/S3) and capturing momentum breaks (R4/S4)
# - Discrete sizing: 0.25 for position, 0.0 for flat
# - Target: 12-30 trades/year (50-120 total over 4 years)

name = "6h_1d_camarilla_volpercentile_ema_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_s4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels for current day)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 6h indicators
    # EMA20 for momentum filter
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume percentile lookback (50 periods ~ 6d6h on 6h)
    volume_pct_lookback = 50
    volume_percentile = pd.Series(volume).rolling(window=volume_pct_lookback, min_periods=volume_pct_lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == volume_pct_lookback else np.nan, raw=False
    ).values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: volume in top 30% percentile (adaptive threshold)
        vol_confirm = volume_percentile[i] > 0.7
        
        # Price position relative to Camarilla levels
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Momentum filter: price relative to EMA20
        price_above_ema = close_price > ema_20[i]
        price_below_ema = close_price < ema_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R4 with volume confirmation and above EMA20
        if close_price > r4 and vol_confirm and price_above_ema:
            enter_long = True
        
        # Short breakout: price breaks below S4 with volume confirmation and below EMA20
        if close_price < s4 and vol_confirm and price_below_ema:
            enter_short = True
        
        # Exit conditions: mean reversion at R3/S3 or EMA crossover
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to R3 or crosses below EMA20
            exit_long = close_price <= r3 or close_price < ema_20[i]
        elif position == -1:
            # Exit short if price rises back to S3 or crosses above EMA20
            exit_short = close_price >= s3 or close_price > ema_20[i]
        
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