#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 12h trend filter and volume confirmation
# - Primary: 4h timeframe for optimal trade frequency (target 20-50/year)
# - HTF: 12h for trend direction (EMA20)
# - Long: Price breaks above H3 Camarilla pivot (resistance) + 12h EMA20 uptrend + volume > 1.5x 20-period MA
# - Short: Price breaks below L3 Camarilla pivot (support) + 12h EMA20 downtrend + volume > 1.5x 20-period MA
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or opposite H4/L4 break
# - Position sizing: 0.25 (discrete level to balance return and risk)
# - No session filter - 4h bars capture global sessions
# - Target: 75-200 total trades over 4 years (19-50/year) - within 4h sweet spot
# - Camarilla pivots work well in ranging markets (common in 2025 BTC/ETH bear/range)
# - 12h EMA20 ensures we trade with intermediate-term trend
# - Volume confirmation on breakout increases reliability
# - Discrete position sizing (0.0, ±0.25) minimizes fee churn from small changes

name = "4h_12h_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 4h Camarilla Pivot Points (based on previous day)
    # Get previous day's OHLC for each 4h bar
    high_1d = get_htf_data(prices, '1d')['high'].values
    low_1d = get_htf_data(prices, '1d')['low'].values
    close_1d = get_htf_data(prices, '1d')['close'].values
    
    # Align daily OHLC to 4h bars
    high_1d_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), close_1d)
    
    # Calculate Camarilla levels for each 4h bar (using previous day's OHLC)
    rng = high_1d_aligned - low_1d_aligned
    h4 = close_1d_aligned + 1.5 * rng
    h3 = close_1d_aligned + 1.25 * rng
    h2 = close_1d_aligned + 1.166 * rng
    h1 = close_1d_aligned + 1.0833 * rng
    pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0
    l1 = close_1d_aligned - 1.0833 * rng
    l2 = close_1d_aligned - 1.166 * rng
    l3 = close_1d_aligned - 1.25 * rng
    l4 = close_1d_aligned - 1.5 * rng
    
    # Calculate 12h EMA(20) for trend direction
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 4h volume moving average (20-period) for volume confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + 12h uptrend + volume spike
            if (close_4h[i] > h3[i] and close_4h[i] > ema_20_12h_aligned[i] and 
                volume_4h[i] > 1.5 * volume_ma_20_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 support + 12h downtrend + volume spike
            elif (close_4h[i] < l3[i] and close_4h[i] < ema_20_12h_aligned[i] and 
                  volume_4h[i] > 1.5 * volume_ma_20_4h[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Pivot Point (mean reversion)
            # 2. Price breaks opposite H4/L4 level (strong reversal)
            # 3. 12h trend changes
            
            if position == 1:  # Long position
                exit_condition = (
                    close_4h[i] < pivot[i] or  # Reverted to pivot
                    close_4h[i] > h4[i] or     # Break above H4 (take profit)
                    close_4h[i] < ema_20_12h_aligned[i]  # 12h trend turned down
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_4h[i] > pivot[i] or  # Reverted to pivot
                    close_4h[i] < l4[i] or     # Break below L4 (take profit)
                    close_4h[i] > ema_20_12h_aligned[i]  # 12h trend turned up
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals