#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w Williams %R mean reversion and volume confirmation
# - Long when KAMA(1d, ER=10) rising AND Williams %R(1w) < -80 (oversold) AND 1d volume > 1.5x 20-bar avg
# - Short when KAMA(1d, ER=10) falling AND Williams %R(1w) > -20 (overbought) AND 1d volume > 1.5x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - KAMA adapts to market noise, reducing whipsaw in ranging markets
# - Williams %R captures short-term exhaustion on weekly timeframe
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_kama_williamsr_volume_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w Williams %R(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1w) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    williams_exit = np.abs(williams_r + 50) < 2.5  # Within 2.5 of -50
    
    # Align 1w Williams %R to 1d timeframe
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1w, williams_oversold)
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1w, williams_overbought)
    williams_exit_aligned = align_htf_to_ltf(prices, df_1w, williams_exit)
    
    # Pre-compute 1d KAMA trend direction (ER=10, fast=2, slow=30)
    close = prices['close'].values
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])).reshape(-1, 1), axis=1)  # simplified
    # Correct ER calculation: ER = |net change| / sum of absolute changes
    net_change = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_change = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(sum_abs_change > 0, net_change / sum_abs_change, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = np.where(np.isnan(sc), 0, sc)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    # Handle first element
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or
            np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or
            np.isnan(williams_exit_aligned[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when KAMA rising AND Williams %R oversold AND volume spike
            if (kama_rising[i] and 
                williams_oversold_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when KAMA falling AND Williams %R overbought AND volume spike
            elif (kama_falling[i] and 
                  williams_overbought_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Williams %R = -50 (mean reversion)
            # Exit when Williams %R returns to equilibrium (-50)
            exit_signal = williams_exit_aligned[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals