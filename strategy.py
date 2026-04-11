#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze + 12h Volume Spike + 1d KAMA Trend Filter
# - Bollinger Band Squeeze: BB Width < 20th percentile indicates low volatility (coiled spring)
# - Entry: Breakout above upper BB (long) or below lower BB (short) with 12h volume spike > 2x average
# - Trend filter: Price must be above/below 1d KAMA to align with higher timeframe trend
# - Exit: Opposite BB breakout or volatility expansion (BB Width > 80th percentile)
# - Works in bull markets (trend continuation breakouts) and bear markets (sharp reversals after squeeze)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits for 4h

name = "4h_12h_1d_bb_squeeze_volume_kama_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for volume spike detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Load 1d data ONCE before loop for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Bollinger Bands (20, 2) on 4h
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    bb_width = (upper_band - lower_band) / basis  # Normalized width
    
    # Pre-compute BB width percentiles for squeeze detection (using expanding window)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.expanding(min_periods=50).quantile(0.2).values  # 20th percentile
    bb_width_pct80 = bb_width_series.expanding(min_periods=50).quantile(0.8).values  # 80th percentile
    
    # Pre-compute 12h volume average (20-period)
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_12h_avg = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_12h_avg_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_avg)
    
    # Pre-compute 1d KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum |close(t) - close(t-1)| over 10 periods
    # Handle array shapes
    change_padded = np.concatenate([[np.nan]*10, change])
    volatility_padded = np.concatenate([[np.nan]*10, volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2 EMA, slow=30 EMA
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i]) or np.isnan(bb_width_pct80[i]) or
            np.isnan(volume_12h_avg_aligned[i]) or np.isnan(kama_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Bollinger Squeeze condition: low volatility (coiled spring)
        squeeze = bb_width[i] < bb_width_pct[i]
        
        # Breakout conditions
        breakout_up = price_high > upper_band[i]  # Break above upper band
        breakout_down = price_low < lower_band[i]  # Break below lower band
        
        # Volume spike: 12h volume > 2x average (confirming institutional interest)
        volume_spike = volume_current > 2.0 * volume_12h_avg_aligned[i]
        
        # KAMA trend filter
        price_above_kama = price_close > kama_aligned[i]
        price_below_kama = price_close < kama_aligned[i]
        
        # Volatility expansion exit condition
        volatility_expansion = bb_width[i] > bb_width_pct80[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Squeeze + upward breakout + volume spike + price above KAMA
        if squeeze and breakout_up and volume_spike and price_above_kama:
            enter_long = True
        
        # Short: Squeeze + downward breakout + volume spike + price below KAMA
        if squeeze and breakout_down and volume_spike and price_below_kama:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on downward breakout OR volatility expansion
            exit_long = breakout_down or volatility_expansion
        elif position == -1:
            # Exit short on upward breakout OR volatility expansion
            exit_short = breakout_up or volatility_expansion
        
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