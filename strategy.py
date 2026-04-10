#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + 1d RSI extremes + 1d volume spike filter
# - Entry: Long when 12h KAMA is rising AND 1d RSI < 30 (oversold) AND 1d volume > 1.5x 20-period average
#          Short when 12h KAMA is falling AND 1d RSI > 70 (overbought) AND 1d volume > 1.5x 20-period average
# - Exit: Close-based reversal - exit long when 12h KAMA starts falling, exit short when 12h KAMA starts rising
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses 12h KAMA for adaptive trend following, daily RSI for mean reversion extremes,
#   and daily volume spike to confirm genuine participation
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 200 total
# - KAMA adapts to market noise, reducing whipsaw in choppy markets
# - RSI extremes work well in both bull and bear markets for mean reversion
# - Volume filter ensures breakouts/mean reversions have participation

name = "12h_1d_kama_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    
    # Pre-compute 1d data
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change_12h = np.abs(np.diff(close_12h, n=10))
    volatility_12h = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)[:len(change_12h)]
    # Pad arrays to match length
    change_12h = np.concatenate([np.full(10, np.nan), change_12h])
    volatility_12h = np.concatenate([np.full(9, np.nan), volatility_12h, np.array([np.nan])])
    
    er_12h = np.where(volatility_12h != 0, change_12h / volatility_12h, 0)
    # Smoothing constants: fastest SC=2/(2+1)=0.667, slowest SC=2/(30+1)=0.0645
    sc_12h = (er_12h * (0.667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama_12h = np.full_like(close_12h, np.nan)
    kama_12h[9] = close_12h[9]  # Start after 10 periods
    for i in range(10, len(close_12h)):
        if not np.isnan(sc_12h[i]):
            kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
        else:
            kama_12h[i] = kama_12h[i-1]
    
    # Calculate 1d RSI (14-period)
    delta_1d = np.diff(close_1d)
    delta_1d = np.concatenate([np.array([np.nan]), delta_1d])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = pd.Series(gain_1d).rolling(window=14, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).rolling(window=14, min_periods=14).mean().values
    rs_1d = np.where(avg_loss_1d != 0, avg_gain_1d / avg_loss_1d, 0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 12h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_1d, kama_12h)  # Wait for 1d bar to close
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close and KAMA
        close_price = close_12h[i]
        kama_value = kama_12h_aligned[i]
        
        # Determine KAMA direction (rising/falling)
        if i > 50:
            kama_prev = kama_12h_aligned[i-1]
            kama_rising = kama_value > kama_prev
            kama_falling = kama_value < kama_prev
        else:
            kama_rising = False
            kama_falling = False
        
        # Volume confirmation: > 1.5x 20-period average
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: KAMA rising AND RSI oversold (<30) AND volume confirmation
            if (kama_rising and 
                rsi_1d_aligned[i] < 30 and 
                volume_confirmation):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA falling AND RSI overbought (>70) AND volume confirmation
            elif (kama_falling and 
                  rsi_1d_aligned[i] > 70 and 
                  volume_confirmation):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when KAMA starts falling
            # Exit short when KAMA starts rising
            if position == 1:
                if kama_falling:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if kama_rising:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals