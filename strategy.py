#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# - Long: Price breaks above Donchian(20) high + 1d ATR(14) > 1.5x 50-period MA of ATR + 1d volume > 1.3x 20-period MA
# - Short: Price breaks below Donchian(20) low + 1d ATR(14) > 1.5x 50-period MA of ATR + 1d volume > 1.3x 20-period MA
# - Exit: Price returns to Donchian(20) midpoint
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag
# - Uses 1d HTF for volatility and volume to ensure breakouts occur with institutional participation
# - Volatility filter ensures we only trade during periods of elevated market activity
# - Works in bull/bear: breakouts in trends with volume/volatility confirmation reduce false signals

name = "4h_1d_donchian_breakout_vol_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) for 4h
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d ATR(14)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 50-period MA of 1d ATR for volatility regime filter
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for ATR14 and ATR MA50)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get aligned 1d data for current 4h bar (completed 1d bar)
        atr_14_current = atr_14_aligned[i]
        atr_ma_50_current = atr_ma_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volatility condition: current 1d ATR > 1.5x 50-period MA of ATR
        vol_condition = atr_14_current > 1.5 * atr_ma_50_current
        
        # Volume spike condition: current 1d volume > 1.3x 20-period MA
        volume_spike = volume_1d_current > 1.3 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high + volatility condition + volume spike
            if (close_price > highest_high[i] and vol_condition and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian(20) low + volatility condition + volume spike
            elif (close_price < lowest_low[i] and vol_condition and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Donchian(20) midpoint
            if position == 1 and close_price <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close_price >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals