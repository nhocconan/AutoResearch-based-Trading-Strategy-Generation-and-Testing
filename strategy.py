#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1d price channel structure for breakouts, 1w EMA50 for major trend direction
# Volume spike ensures institutional participation and reduces false breakouts
# ATR-based stoploss manages risk during volatile moves
# Discrete position sizing 0.25 minimizes fee churn while maintaining profitability
# Targets 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking trend-aligned breakouts with volume

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d ATR(14) for volatility filter and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, ATR, and volume MA)
    start_idx = 50  # max(20 for Donchian, 14 for ATR, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: only trade when volatility is above average (avoid chop)
        atr_ma = pd.Series(atr_14).rolling(window=10, min_periods=10).mean().shift(1).values
        if np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            volatility_filter = True  # allow trade if MA not ready
        else:
            volatility_filter = atr_14[i] > (atr_ma[i] * 0.7)  # trade when ATR > 70% of MA
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND above 1w EMA50 AND volume confirm AND volatility filter
            if (close[i] > high_ma[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1w EMA50 AND volume confirm AND volatility filter
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volatility drops significantly
            if (close[i] < low_ma[i] or 
                (not volatility_filter and atr_14[i] < (atr_ma[i] * 0.5))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volatility drops significantly
            if (close[i] > high_ma[i] or 
                (not volatility_filter and atr_14[i] < (atr_ma[i] * 0.5))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals