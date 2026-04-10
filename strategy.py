#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d ATR(14) rising AND volume > 1.8x 20-bar avg
# - Short when price breaks below 20-period Donchian low AND 1d ATR(14) rising AND volume > 1.8x 20-bar avg
# - Exit when price crosses 10-period EMA (adaptive trailing stop)
# - Uses 1d ATR for volatility filter to avoid whipsaw in low volatility
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 25-35 trades/year on 4h timeframe (100-140 total over 4 years)
# - Donchian breakouts capture momentum; volatility filter ensures breakouts occur in expanding volatility

name = "4h_1d_donchian_breakout_volume_volatility_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - exponential moving average of TR
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR rising: current > previous
    atr_rising = np.zeros_like(atr14_1d, dtype=bool)
    atr_rising[1:] = atr14_1d[1:] > atr14_1d[:-1]
    
    # Align HTF ATR and rising flag to LTF
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising.astype(float))
    
    # Pre-compute Donchian channels (20-period) on LTF
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute 10-period EMA for exit
    ema10 = prices['close'].ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(atr14_1d_aligned[i]) or np.isnan(atr_rising_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema10[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND ATR rising with volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                atr_rising_aligned[i] > 0.5 and  # ATR is rising
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND ATR rising with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  atr_rising_aligned[i] > 0.5 and  # ATR is rising
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when price crosses 10 EMA
            # Exit when price crosses 10-period EMA (adaptive trailing stop)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < ema10[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > ema10[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals