#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike
# - Long when price breaks above 20-period high AND ATR(14) < ATR(50) AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-period low AND ATR(14) < ATR(50) AND volume > 1.5x 20-bar avg
# - Exit when price crosses 10-period EMA in opposite direction
# - Uses 1d ATR regime filter to avoid high-volatility choppy markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 30-50 trades/year on 4h timeframe (120-200 total over 4 years)
# - Donchian breakouts capture momentum; ATR filter ensures breakouts occur in lowering volatility (often precedes strong moves)

name = "4h_1d_donchian_breakout_atr_filter_volume_v1"
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
    
    # Pre-compute 1d ATR regime filter: ATR(14) < ATR(50) indicates lowering volatility
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
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_regime = atr14 < atr50  # True when short-term ATR < long-term ATR (lowering vol)
    
    # Align HTF ATR regime to LTF
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Pre-compute Donchian channels (20-period) from LTF data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 10-period EMA for exit
    close_s = prices['close']
    ema10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema10[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above 20-period high AND ATR regime favorable AND volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                atr_regime_aligned[i] > 0.5 and  # ATR regime true (as float)
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below 20-period low AND ATR regime favorable AND volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  atr_regime_aligned[i] > 0.5 and  # ATR regime true (as float)
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses 10-period EMA in opposite direction
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