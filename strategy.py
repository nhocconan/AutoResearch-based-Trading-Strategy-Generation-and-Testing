#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long: Williams %R(14) < -80 (oversold) + price > 1d EMA50 (uptrend) + 1d volume > 1.2x 20-period MA
# - Short: Williams %R(14) > -20 (overbought) + price < 1d EMA50 (downtrend) + 1d volume > 1.2x 20-period MA
# - Exit: Williams %R crosses above -50 for longs, below -50 for shorts
# - Position sizing: 0.25 (discrete level)
# - Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag
# - Williams %R captures short-term reversals in ranging markets, EMA50 filters for trend alignment,
#   volume confirms conviction. Works in both bull/bear: mean reversion in ranges, trend filter avoids counter-trend trades.

name = "6h_1d_williamsr_meanreversion_vol_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) for 6h
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for EMA50)
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get aligned 1d data for current 6h bar (completed 1d bar)
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume spike condition: current 1d volume > 1.2x 20-period MA
        volume_spike = volume_1d_current > 1.2 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + price > 1d EMA50 + volume spike
            if (williams_r[i] < -80 and close_price > ema_50_current and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + price < 1d EMA50 + volume spike
            elif (williams_r[i] > -20 and close_price < ema_50_current and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when Williams %R crosses above -50
            if position == 1 and williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            # Exit short when Williams %R crosses below -50
            elif position == -1 and williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals