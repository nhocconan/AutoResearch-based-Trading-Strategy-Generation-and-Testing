#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and ATR regime filter
# - Donchian(20) breakout on 4h captures strong momentum moves
# - 1d volume spike filter (volume > 2.0x 20-day average) confirms institutional participation
# - ATR regime filter: only trade when ATR(14) < ATR(50) (low volatility environment) to avoid chop
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Exit: opposite Donchian breakout or volatility expansion (ATR(14) > 1.5x ATR(50))
# - Works in both bull/bear: Donchian breakouts capture trends, volume filter reduces false signals,
#   ATR regime filter avoids whipsaws in ranging markets
# - Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag

name = "4h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1d ATR regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_14 < atr_50  # Low volatility regime
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Pre-compute 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR for volatility expansion exit
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1)) if 'close_4h' in locals() else np.abs(high_4h - np.roll(prices['close'].values, 1))
    tr_4h3 = np.abs(low_4h - np.roll(prices['close'].values, 1))
    tr_4h1[0] = high_4h[0] - low_4h[0]
    tr_4h2[0] = 0
    tr_4h3[0] = 0
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_50_4h = pd.Series(tr_4h).rolling(window=50, min_periods=50).mean().values
    vol_expansion = atr_14_4h > (1.5 * atr_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current close for breakout detection
        close_price = prices['close'].iloc[i]
        
        # Breakout conditions
        bullish_breakout = close_price > donchian_high[i]
        bearish_breakout = close_price < donchian_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish breakout AND volume spike AND low volatility regime
            if bullish_breakout and volume_spike_aligned[i] and atr_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish breakout AND volume spike AND low volatility regime
            elif bearish_breakout and volume_spike_aligned[i] and atr_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: opposite breakout OR volatility expansion
            exit_long = bearish_breakout or vol_expansion[i]
            exit_short = bullish_breakout or vol_expansion[i]
            
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals