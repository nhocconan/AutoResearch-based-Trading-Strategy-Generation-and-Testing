#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume Spike Reversal with 1d EMA50 Trend Filter
# Long when: volume > 2.5x 20-bar avg AND close < 1d EMA50 AND RSI(14) < 30 (oversold bounce)
# Short when: volume > 2.5x 20-bar avg AND close > 1d EMA50 AND RSI(14) > 70 (overbought rejection)
# Exit: reverse signal or RSI returns to 50 (mean reversion)
# Uses discrete sizing (0.25) and volume spike confirmation to reduce false signals.
# Target: 15-35 trades/year on 12h timeframe. Works in bull/bear via mean reversion from extremes.

name = "12h_VolumeSpike_RSI_Reversal_1dEMA50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 12h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: >2.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # volume MA and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_rsi = rsi_values[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: RSI returns to 50 (mean reversion) or reverse signal
            if curr_rsi >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to 50 (mean reversion) or reverse signal
            if curr_rsi <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when volume spike AND price below 1d EMA50 AND RSI oversold (<30)
            if vol_spike and curr_close < curr_ema50 and curr_rsi < 30:
                signals[i] = 0.25
                position = 1
            # Short when volume spike AND price above 1d EMA50 AND RSI overbought (>70)
            elif vol_spike and curr_close > curr_ema50 and curr_rsi > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals