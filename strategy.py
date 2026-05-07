# 12H_RSI_Overbought_Oversold_MeanReversion_v1
# Hypothesis: Use 12h RSI for mean reversion signals. Long when RSI < 30 (oversold) and price above 12h EMA50 (uptrend filter). Short when RSI > 70 (overbought) and price below 12h EMA50 (downtrend filter). Volume confirmation: current volume > 1.5x 20-period average volume. Designed to work in both bull and bear markets by combining momentum extremes with trend filter.
# Timeframe: 12h
# Expected trades: 15-25 per year (60-100 total over 4 years) to stay within fee-efficient range.
name = "12H_RSI_Overbought_Oversold_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h RSI(14)
    close_12h = pd.Series(df_12h['close'])
    delta = close_12h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 12h EMA50 for trend filter
    ema50 = close_12h.ewm(span=50, adjust=False).mean().values
    
    # Align indicators to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = 50  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (2 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: RSI < 30 (oversold) and price above EMA50 (uptrend)
            if (rsi_aligned[i] < 30 and close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: RSI > 70 (overbought) and price below EMA50 (downtrend)
            elif (rsi_aligned[i] > 70 and close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60)
            if position == 1 and rsi_aligned[i] > 40:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and rsi_aligned[i] < 60:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals