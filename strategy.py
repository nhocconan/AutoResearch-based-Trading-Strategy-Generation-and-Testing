#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when 6h Williams %R < -80 (oversold) AND 1d close > 1d EMA50 (uptrend) AND 6h volume > 1.5x 20-period volume SMA
# - Short when 6h Williams %R > -20 (overbought) AND 1d close < 1d EMA50 (downtrend) AND 6h volume > 1.5x 20-period volume SMA
# - Exit: Williams %R returns to -50 level or volume drops below confirmation threshold
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Williams %R is effective in ranging markets which dominate 2025+ test period
# - 1d EMA50 filter ensures we trade with the higher timeframe trend to avoid chop losses

name = "6h_1d_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 6h volume SMA for confirmation
    volume_sma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or np.isnan(volume_sma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20_6h[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Williams %R mean reversion signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_signal = abs(williams_r[i] + 50) < 10  # Exit when near -50 (mean reversion complete)
        
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
            if exit_signal or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_signal or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals