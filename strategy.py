#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1d close > 1d EMA50 (bullish trend)
# - Short when Williams %R(14) > -20 (overbought) AND 1d close < 1d EMA50 (bearish trend)
# - Volume confirmation: 6h volume > 1.3x 20-period volume SMA
# - Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits

name = "6h_1d_williamsr_meanreversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 6h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Williams %R signals
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        wr_exit_long = williams_r[i] > -50  # Exit long when crosses above -50
        wr_exit_short = williams_r[i] < -50  # Exit short when crosses below -50
        
        if position == 0:  # Flat - look for entry
            if wr_oversold and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif wr_overbought and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if wr_exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if wr_exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals