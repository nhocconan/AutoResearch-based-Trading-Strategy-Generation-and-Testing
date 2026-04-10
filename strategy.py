#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ATR-based volatility filter and volume confirmation
# - Long: Price breaks above Donchian(20) high + 1w ATR(14) > 1.5x 50-period MA of ATR + 1w volume > 1.3x 20-period MA
# - Short: Price breaks below Donchian(20) low + 1w ATR(14) > 1.5x 50-period MA of ATR + 1w volume > 1.3x 20-period MA
# - Exit: Price returns to Donchian(20) midpoint
# - Position sizing: 0.25 (discrete level)
# - Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and fee drag
# - Uses 1w HTF for volatility and volume to ensure breakouts occur with institutional participation
# - Volatility filter ensures we only trade during periods of elevated market activity
# - Works in bull/bear: breakouts in trends with volume/volatility confirmation reduce false signals

name = "1d_1w_donchian_breakout_vol_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute daily OHLCV
    open_d = prices['open'].values
    high_d = prices['high'].values
    low_d = prices['low'].values
    close_d = prices['close'].values
    volume_d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian(20) for daily
    highest_high = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1w ATR(14)
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate 50-period MA of 1w ATR for volatility regime filter
    atr_ma_50_1w = pd.Series(atr_14_1w).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_50_1w)
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for ATR14 and ATR MA50)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current daily close
        close_price = close_d[i]
        
        # Get aligned 1w data for current daily bar (completed 1w bar)
        atr_14_current = atr_14_aligned[i]
        atr_ma_50_current = atr_ma_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        
        # Volatility condition: current 1w ATR > 1.5x 50-period MA of ATR
        vol_condition = atr_14_current > 1.5 * atr_ma_50_current
        
        # Volume spike condition: current 1w volume > 1.3x 20-period MA
        volume_spike = volume_1w_current > 1.3 * volume_ma_current
        
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