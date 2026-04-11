#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w ATR regime filter
# - Long: price breaks above 20-period Donchian high, volume > 2.0x 20-period avg, 1w ATR(14) > 0.5x 50-period MA
# - Short: price breaks below 20-period Donchian low, volume > 2.0x 20-period avg, 1w ATR(14) > 0.5x 50-period MA
# - Exit: price returns to opposite Donchian level or ATR-based stop (2x ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves; volume spike confirms institutional interest
# - 1w ATR regime ensures sufficient volatility for breakout validity, reducing false signals

name = "12h_1d_1w_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return signals
    
    # Pre-compute 1w ATR(14) and 50-period MA for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # ATR(14)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 50-period MA of ATR
    atr_ma_50_1w = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: current ATR > 0.5x 50-period MA (sufficient volatility)
    atr_regime_1w = atr_1w > 0.5 * atr_ma_50_1w
    
    # Align 1w ATR regime to 12h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1w, atr_regime_1w)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels (20-period) on 12h timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Volatility regime: 1w ATR regime indicates sufficient volatility
        vol_regime = atr_regime_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above upper Donchian, volume confirmation, volatility regime
        if close_price > upper_channel and vol_confirm and vol_regime:
            enter_long = True
        
        # Short breakout: price below lower Donchian, volume confirmation, volatility regime
        if close_price < lower_channel and vol_confirm and vol_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches lower channel or ATR-based stop
            exit_long = (close_price <= lower_channel) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price reaches upper channel or ATR-based stop
            exit_short = (close_price >= upper_channel) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals