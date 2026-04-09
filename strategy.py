#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Uses 1d EMA(50) for trend direction (bullish if price > EMA50, bearish if price < EMA50)
# - Enters long in bullish trend when 6h Williams %R(14) < -80 (oversold) + volume > 1.5 * 20-period average
# - Enters short in bearish trend when 6h Williams %R(14) > -20 (overbought) + volume > 1.5 * 20-period average
# - Exits when Williams %R reverts to midpoint (-50) or opposite extreme
# - Position size: 0.25 (25% of capital) to limit drawdown in 2022-like crashes
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging/mean-reverting markets (2022-2024) and catches reversals in trends

name = "6h_1d_williamsr_meanrev_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Pre-compute volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion to midpoint or overbought
            if williams_r[i] >= -50:  # Reverted to midpoint
                position = 0
                signals[i] = 0.0
            elif williams_r[i] > -20:  # Overbought exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion to midpoint or oversold
            if williams_r[i] <= -50:  # Reverted to midpoint
                position = 0
                signals[i] = 0.0
            elif williams_r[i] < -80:  # Oversold exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with trend filter and volume confirmation
            if close[i] > ema_50_aligned[i]:  # Bullish trend
                if williams_r[i] < -80 and volume_confirm[i]:  # Oversold in bullish trend
                    position = 1
                    signals[i] = 0.25
            elif close[i] < ema_50_aligned[i]:  # Bearish trend
                if williams_r[i] > -20 and volume_confirm[i]:  # Overbought in bearish trend
                    position = -1
                    signals[i] = -0.25
    
    return signals