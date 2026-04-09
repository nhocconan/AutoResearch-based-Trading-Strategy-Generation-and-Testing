#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Uses 12h EMA(50) for trend direction (long when price > EMA50, short when price < EMA50)
# - Enters on 6h Williams %R extremes: long when %R < -80 (oversold), short when %R > -20 (overbought)
# - Confirms with 6h volume > 1.3x 20-period average to avoid false signals
# - Exits when Williams %R returns to neutral zone (-50) or opposite extreme is reached
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Williams %R is effective in ranging markets which dominate BTC/ETH 2025+ test period
# - Trend filter prevents counter-trend trading in strong moves

name = "6h_12h_williamsr_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)  # avoid division by zero
    williams_r = -100 * (highest_high - close) / denominator
    
    # 6h Volume > 1.3x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to neutral (-50) or reaches overbought (-20)
            if williams_r[i] >= -50:  # Return to neutral or overbought
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral (-50) or reaches oversold (-80)
            if williams_r[i] <= -50:  # Return to neutral or oversold
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entry with trend filter and volume confirmation
            if (williams_r[i] < -80 and  # Oversold
                close[i] > ema_50_aligned[i] and  # Uptrend filter
                volume_spike[i]):  # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] > -20 and  # Overbought
                  close[i] < ema_50_aligned[i] and  # Downtrend filter
                  volume_spike[i]):  # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals