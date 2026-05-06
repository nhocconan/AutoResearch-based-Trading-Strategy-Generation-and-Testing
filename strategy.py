#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Uses 60/100 period SMAs from Williams Alligator to define trend (Green > Red > Blue)
# Confirms with 1d EMA50 trend alignment to avoid counter-trend trades
# Volume spike (>2.0x 20-bar average) ensures breakout strength
# Designed for low-frequency trading (target: 60-120 total trades over 4 years) to minimize fee drag
# Works in bull/bear markets: Alligator identifies trends, volume confirms strength, 1d filter reduces whipsaw

name = "6h_WilliamsAlligator_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 60/100 period SMAs
    # Jaw (blue): 130-period SMMA of median price, shifted 8 bars ahead
    # Teeth (red): 80-period SMMA of median price, shifted 5 bars ahead
    # Lips (green): 50-period SMMA of median price, shifted 3 bars ahead
    median_price = (high + low) / 2
    
    # Calculate SMAs (using SMA as proxy for SMMA)
    sma_130 = pd.Series(median_price).rolling(window=130, min_periods=130).mean().values
    sma_80 = pd.Series(median_price).rolling(window=80, min_periods=80).mean().values
    sma_50 = pd.Series(median_price).rolling(window=50, min_periods=50).mean().values
    
    # Shift to align with Alligator rules
    jaw = np.roll(sma_130, 8)  # 130-period shifted 8 bars
    teeth = np.roll(sma_80, 5)  # 80-period shifted 5 bars
    lips = np.roll(sma_50, 3)   # 50-period shifted 3 bars
    
    # Fill NaN from rolling and shifts
    jaw = np.where(np.isnan(jaw), median_price, jaw)
    teeth = np.where(np.isnan(teeth), median_price, teeth)
    lips = np.where(np.isnan(lips), median_price, lips)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(130, n):  # Start after Alligator warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price > EMA50 (1d uptrend) AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price < EMA50 (1d downtrend) AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit: Alligator turns bearish OR price retraces 30% of ATR from extreme
            if lips[i] < teeth[i] or close[i] <= long_extreme - 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit: Alligator turns bullish OR price retraces 30% of ATR from extreme
            if lips[i] > teeth[i] or close[i] >= short_extreme + 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals