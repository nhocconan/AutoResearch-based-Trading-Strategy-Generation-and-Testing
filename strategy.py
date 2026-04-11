#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and 1d chop regime filter
# - Long: Price breaks above Donchian(20) upper band + volume > 1.5x 20-period average + chop < 61.8 (trending regime)
# - Short: Price breaks below Donchian(20) lower band + volume > 1.5x 20-period average + chop < 61.8 (trending regime)
# - Exit: Price returns to Donchian midpoint or ATR-based stop (2.0 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves effectively in both bull and bear markets
# - Volume confirmation ensures institutional participation and reduces false breakouts
# - Chop regime filter (CHOP < 61.8) ensures we only trade in trending markets, avoiding whipsaw in ranging conditions

name = "4h_1d_donchian_volume_chop_v3"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Chop Index (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop Index formula: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero when HH == LL
    chop_denominator = hh_14 - ll_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Small epsilon to avoid div by zero
    chop_raw = 100 * np.log10(tr_sum_14 / chop_denominator) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50.0, chop_raw)  # Default to 50 if invalid
    
    # Align 1d Chop Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels (20-period) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close_price > highest_high[i]
        breakout_down = close_price < lowest_low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Regime filter: chop < 61.8 indicates trending market (avoid ranging/choppy conditions)
        chop_filter = chop_aligned[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long entry: bullish breakout with volume and trending regime
        if breakout_up and vol_confirm and chop_filter:
            enter_long = True
        
        # Short entry: bearish breakout with volume and trending regime
        if breakout_down and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Donchian midpoint or ATR-based stop
            exit_long = (close_price <= donchian_mid[i]) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to Donchian midpoint or ATR-based stop
            exit_short = (close_price >= donchian_mid[i]) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
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