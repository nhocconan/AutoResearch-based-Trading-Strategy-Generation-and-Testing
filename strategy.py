#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1d HTF - 6h Bollinger Breakout with 1d Volume Regime Filter
    # Designed to capture volatility expansion moves in both bull and bear markets
    # Volume regime filter avoids low-volume false breakouts; Bollinger bands provide dynamic support/resistance
    # Target: 50-150 total trades over 4 years (12-37/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF volume regime and Bollinger context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 6h Bollinger Bands (20-period, 2 std)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Calculate 1d Volume 20-period average for regime filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1d volume > 1.2x 20-period average (avoid low-volume breakouts)
        # We need to map current 6h bar to its corresponding 1d bar for volume check
        # Since we aligned the 1d volume MA, we can use it directly
        vol_regime = volume_1d[i // 24] > 1.2 * vol_ma_20_1d_aligned[i] if i // 24 < len(volume_1d) else False
        
        # Bollinger breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Entry conditions: breakout + volume regime
        enter_long = breakout_up and vol_regime
        enter_short = breakout_down and vol_regime
        
        # Exit conditions: return to Bollinger middle (mean reversion)
        exit_long = position == 1 and close[i] <= sma_20[i]
        exit_short = position == -1 and close[i] >= sma_20[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_bollinger_breakout_volume_regime_v1"
timeframe = "6h"
leverage = 1.0