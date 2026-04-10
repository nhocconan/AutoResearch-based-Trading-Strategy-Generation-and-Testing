#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d volume spike and 1w trend filter
# - Long when Williams %R(14) < -80 (oversold) AND 1d volume > 1.5x 20-period volume SMA AND 1w close > 1w EMA20
# - Short when Williams %R(14) > -20 (overbought) AND 1d volume > 1.5x 20-period volume SMA AND 1w close < 1w EMA20
# - Exit: Williams %R returns to -50 level or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Williams %R identifies exhaustion points, volume spike confirms conviction, 1w EMA filters trend

name = "6h_1d_1w_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 6h volume for confirmation
    volume_sma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(close_1w_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(volume_sma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_6h = volume[i] > 1.5 * volume_sma_20_6h[i]
        vol_confirm_1d = volume_1d[i // 4] > 1.5 * volume_sma_20_1d_aligned[i] if i // 4 < len(volume_1d) else False
        vol_confirm = vol_confirm_6h and vol_confirm_1d
        
        # Trend filter: 1w close vs 1w EMA20
        trend_bullish = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Williams %R signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_level = abs(williams_r[i] + 50) < 10  # Near -50 level
        
        if position == 0:  # Flat - look for entry
            if oversold and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif overbought and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_level or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_level or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals