#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load weekly data (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate ATR on daily (14-period)
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR on weekly (14-period)
    tr_1w = np.maximum(high_1w - low_1w,
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align HTF data to daily timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_1d[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        
        # Volatility regime: low volatility when weekly ATR is not elevated
        vol_regime_low = atr_1w_val <= (atr_1d_val * 1.5)
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly EMA(34) OR volatility regime shifts to high
            if (price < ema_34_1w_val) or (not vol_regime_low):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly EMA(34) OR volatility regime shifts to high
            if (price > ema_34_1w_val) or (not vol_regime_low):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above weekly EMA(34) AND low volatility regime
            if (price > ema_34_1w_val) and vol_regime_low:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below weekly EMA(34) AND low volatility regime
            elif (price < ema_34_1w_val) and vol_regime_low:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_EMA34_VolRegime_Filter"
timeframe = "1d"
leverage = 1.0