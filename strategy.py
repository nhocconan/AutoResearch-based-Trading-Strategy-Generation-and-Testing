#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# - Long when price breaks above Donchian upper (20-period high) AND ATR(14) > ATR(50) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian lower (20-period low) AND ATR(14) > ATR(50) AND volume > 1.5x 20-bar avg
# - Exit when price crosses opposite Donchian band (full reversal)
# - Uses 1d ATR ratio to ensure breakouts occur in sufficient volatility regimes
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years)
# - Donchian breakouts capture strong momentum moves; volatility filter avoids choppy markets

name = "4h_1d_donchian_breakout_atr_volume_v1"
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
    
    # Pre-compute Donchian channels from 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20): 20-period high/low
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d ATR(14) and ATR(50) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = high_1d[0] - close_1d[0]  # first bar
    tr3[0] = high_1d[0] - close_1d[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14 / (atr50 + 1e-10)  # avoid division by zero
    
    # Align HTF ATR ratio to LTF
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND sufficient volatility AND volume spike
            if (prices['close'].iloc[i] > donch_high[i] and 
                atr_ratio_aligned[i] > 1.2 and  # ATR(14) > 1.2 * ATR(50)
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND sufficient volatility AND volume spike
            elif (prices['close'].iloc[i] < donch_low[i] and 
                  atr_ratio_aligned[i] > 1.2 and  # ATR(14) > 1.2 * ATR(50)
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on opposite band break
            # Exit when price crosses opposite Donchian band (full reversal)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < donch_low[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > donch_high[i]:
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