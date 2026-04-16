# 1D Mean Reversion Strategy with RSI and Volume Confirmation
# Uses 1D timeframe with mean reversion on RSI extremes, filtered by volume
# Works in both bull and bear markets by focusing on reversals rather than trends
# Entry: RSI < 30 (oversold) or RSI > 70 (overbought) with volume spike
# Exit: RSI returns to neutral zone (40-60)
# Position size: 0.25 for mean reversion trades
# Volume filter reduces false signals and improves win rate
# Simple design avoids overtrading while capturing mean reversion opportunities

#!/usr/bin/env python3
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
    
    # === 1D DATA (PRIMARY TIMEFRAME) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # RSI(14) for mean reversion signals
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # EXIT LOGIC: Return to neutral RSI zone
        if position == 1:  # Long position
            if rsi_val >= 40:  # Exit when RSI returns to neutral
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            if rsi_val <= 60:  # Exit when RSI returns to neutral
                signals[i] = 0.0
                position = 0
                continue
        
        # ENTRY LOGIC (only when flat)
        if position == 0:
            # LONG: RSI oversold with volume confirmation
            if (rsi_val < 30) and (vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: RSI overbought with volume confirmation
            elif (rsi_val > 70) and (vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1D_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0