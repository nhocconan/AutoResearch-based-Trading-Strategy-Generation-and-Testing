#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# - Long: Williams %R(14) < -80 (oversold) + 1d close > 1d EMA50 (bullish trend) + 6h volume > 2.0x 20-period average volume
# - Short: Williams %R(14) > -20 (overbought) + 1d close < 1d EMA50 (bearish trend) + 6h volume > 2.0x 20-period average volume
# - Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-160 total trades over 4 years (20-40/year) to stay within HARD MAX: 300 total
# - Williams %R captures short-term extremes, 1d EMA50 ensures trend alignment, volume spike confirms institutional participation

name = "6h_1d_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for Williams %R calculation
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data for EMA50
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) for 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume moving average (20-period)
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for volume MA)
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        williams_r_current = williams_r[i]
        close_price = close_6h[i]
        ema_50_current = ema_50_aligned[i]
        volume_current = volume_6h[i]
        volume_ma_current = volume_ma_20_6h[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmation = volume_current > 2.0 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold + bullish 1d trend + volume spike
            if (williams_r_current < -80 and 
                close_price > ema_50_current and 
                volume_confirmation):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought + bearish 1d trend + volume spike
            elif (williams_r_current > -20 and 
                  close_price < ema_50_current and 
                  volume_confirmation):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            if position == 1:  # Long position
                # Exit when Williams %R crosses above -50 (mean reversion complete)
                if williams_r_current > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit when Williams %R crosses below -50 (mean reversion complete)
                if williams_r_current < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals