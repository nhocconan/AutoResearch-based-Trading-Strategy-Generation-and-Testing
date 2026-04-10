#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) on 6h: oversold < -80 for long, overbought > -20 for short
# - 1d trend filter: price > EMA50 for long bias, price < EMA50 for short bias
# - Volume confirmation: 6h volume > 1.5x 20-period MA to ensure participation
# - Exit: Williams %R returns to -50 (mean reversion midpoint) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and fees
# - Works in bull/bear: mean reversion in ranges, trend filter prevents counter-trend in strong moves

name = "6h_1d_williamsr_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) for 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h (using prior completed 1d bar)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h volume moving average (20-period)
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for Williams %R and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h data
        close_price = close_6h[i]
        williams_r_current = williams_r[i]
        ema50_current = ema50_aligned[i]
        volume_6h_current = volume_6h[i]
        volume_ma_current = volume_ma_20_6h[i]
        
        # Volume spike condition: current 6h volume > 1.5x 20-period MA
        volume_spike = volume_6h_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + price > EMA50 (uptrend bias) + volume spike
            if (williams_r_current < -80 and close_price > ema50_current and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + price < EMA50 (downtrend bias) + volume spike
            elif (williams_r_current > -20 and close_price < ema50_current and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R returns to -50 (mean reversion midpoint) or opposite signal
            if position == 1 and williams_r_current >= -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r_current <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals