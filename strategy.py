#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and 1d volume confirmation
# - Entry: Long when Williams %R(14) < -80 (oversold) + 1w EMA21 uptrend + 1d volume > 1.5x 20-period average
#          Short when Williams %R(14) > -20 (overbought) + 1w EMA21 downtrend + 1d volume > 1.5x 20-period average
# - Exit: Close-based reversal - exit long when Williams %R > -50, exit short when Williams %R < -50
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Williams %R for mean reversion signals, weekly EMA for trend filter, daily volume for confirmation
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 200 total
# - Williams %R identifies overextended moves likely to reverse, weekly trend ensures we trade with higher timeframe momentum,
#   volume confirmation reduces false signals from low participation breakouts

name = "12h_1w_1d_williamsr_meanrev_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Pre-compute 1w data for indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pre-compute 1d data for indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA21 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros_like(close_1d)
    denominator = highest_high_14 - lowest_low_14
    mask = (denominator != 0) & (~np.isnan(denominator))
    williams_r[mask] = ((highest_high_14[mask] - close_1d[mask]) / denominator[mask]) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Volume confirmation: > 1.5x 20-period average
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + weekly uptrend + volume confirmation
            if (williams_r_aligned[i] < -80 and 
                close_price > ema_21_1w_aligned[i] and 
                volume_confirmation):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + weekly downtrend + volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  close_price < ema_21_1w_aligned[i] and 
                  volume_confirmation):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when Williams %R > -50 (recovering from oversold)
            # Exit short when Williams %R < -50 (declining from overbought)
            if position == 1:
                if williams_r_aligned[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_aligned[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals