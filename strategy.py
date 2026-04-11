#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
# - Long when price breaks above Camarilla H3 level with volume > 1.8x 20-day average and weekly close > weekly open (bullish week)
# - Short when price breaks below Camarilla L3 level with volume > 1.8x 20-day average and weekly close < weekly open (bearish week)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits for 1d
# - Volume confirmation ensures high-conviction breakouts
# - Weekly trend filter aligns with higher timeframe momentum to avoid counter-trend trades
# - Works in both bull (breakouts with volume in bullish weeks) and bear (breakdowns with volume in bearish weeks)

name = "1d_1w_camarilla_pivot_volume_trend_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # H3 = Pivot + (H - L) * 1.1 / 4
    # L3 = Pivot - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    camarilla_h3 = pivot_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_l3 = pivot_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_h3_shifted = np.roll(camarilla_h3, 1)
    camarilla_l3_shifted = np.roll(camarilla_l3, 1)
    camarilla_h3_shifted[0] = np.nan  # First value invalid
    camarilla_l3_shifted[0] = np.nan
    
    # 1d volume SMA (20-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (identity alignment but using helper for consistency)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_shifted)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_shifted)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        # Fallback to neutral trend if insufficient weekly data
        weekly_bullish = np.ones(n, dtype=bool)
        weekly_bearish = np.ones(n, dtype=bool)
    else:
        # Weekly trend: bullish if weekly close > weekly open, bearish if close < open
        weekly_open = df_1w['open'].values
        weekly_close = df_1w['close'].values
        weekly_bullish_raw = weekly_close > weekly_open
        weekly_bearish_raw = weekly_close < weekly_open
        
        # Align weekly trend to daily timeframe
        weekly_bullish = align_htf_to_ltf(prices, df_1w, weekly_bullish_raw.astype(float)) > 0.5
        weekly_bearish = align_htf_to_ltf(prices, df_1w, weekly_bearish_raw.astype(float)) > 0.5
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-day average
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H3 + volume confirmation + bullish week
        if (price_close > camarilla_h3_aligned[i] and 
            vol_confirm and 
            weekly_bullish[i]):
            enter_long = True
        
        # Short: Price breaks below Camarilla L3 + volume confirmation + bearish week
        if (price_close < camarilla_l3_aligned[i] and 
            vol_confirm and 
            weekly_bearish[i]):
            enter_short = True
        
        # Exit conditions: opposite Camarilla level break or volume collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR volume drops below average
            exit_long = (price_close < camarilla_l3_aligned[i]) or (volume_current < volume_sma_20_aligned[i])
        elif position == -1:
            # Exit short if price breaks above H3 OR volume drops below average
            exit_short = (price_close > camarilla_h3_aligned[i]) or (volume_current < volume_sma_20_aligned[i])
        
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