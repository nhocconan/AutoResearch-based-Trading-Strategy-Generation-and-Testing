#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot Breakout with 1w trend filter and volume confirmation
# - Primary: 1d timeframe for lower trade frequency and better risk/reward
# - HTF: 1w for trend direction (close above/below weekly EMA21)
# - Long: Price breaks above H3 Camarilla pivot (resistance) + weekly close > weekly EMA21 + volume > 1.5x 20-day MA
# - Short: Price breaks below L3 Camarilla pivot (support) + weekly close < weekly EMA21 + volume > 1.5x 20-day MA
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or opposite H4/L4 break
# - Position sizing: 0.25 (discrete level to balance profit and fee drag)
# - Target: 30-100 total trades over 4 years (7-25/year) - within 1d sweet spot
# - Camarilla pivots work well in ranging markets (common in 2025 BTC/ETH bear/range)
# - Weekly EMA21 ensures we trade with intermediate-term trend
# - Volume confirmation on breakout increases reliability
# - This strategy avoids overtrading by using daily timeframe and tight entry conditions

name = "1d_1w_camarilla_pivot_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla Pivot Points (based on previous day)
    # We need to align daily OHLC to daily bars (shift by 1 to use previous day)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to nan since there's no previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for each 1d bar (using previous day's OHLC)
    rng = high_1d_prev - low_1d_prev
    h3 = close_1d_prev + 1.25 * rng
    l3 = close_1d_prev - 1.25 * rng
    h4 = close_1d_prev + 1.5 * rng
    l4 = close_1d_prev - 1.5 * rng
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    
    # Calculate weekly EMA(21) for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(volume_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + weekly uptrend + volume spike
            if (close_1d[i] > h3[i] and close_1d[i] > ema_21_1w_aligned[i] and 
                volume_1d[i] > 1.5 * volume_ma_20_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 support + weekly downtrend + volume spike
            elif (close_1d[i] < l3[i] and close_1d[i] < ema_21_1w_aligned[i] and 
                  volume_1d[i] > 1.5 * volume_ma_20_1d[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Pivot Point (mean reversion)
            # 2. Price breaks opposite H4/L4 level (strong reversal)
            # 3. Weekly trend changes
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < pivot[i] or  # Reverted to pivot
                    close_1d[i] > h4[i] or     # Break above H4 (take profit)
                    close_1d[i] < ema_21_1w_aligned[i]  # Weekly trend turned down
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > pivot[i] or  # Reverted to pivot
                    close_1d[i] < l4[i] or     # Break below L4 (take profit)
                    close_1d[i] > ema_21_1w_aligned[i]  # Weekly trend turned up
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals