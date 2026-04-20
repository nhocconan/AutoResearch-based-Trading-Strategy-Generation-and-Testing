#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WickReversal_Pullback_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily data
    open_d = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend: 20-week EMA on weekly closes
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume ratio (current vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        open_val = open_d[i]
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        ema_val = ema_20_1w_aligned[i]
        atr_val = atr[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(open_val) or np.isnan(high_val) or np.isnan(low_val) or 
            np.isnan(close_val) or np.isnan(ema_val) or np.isnan(atr_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily range and wick sizes
        daily_range = high_val - low_val
        if daily_range <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        upper_wick = high_val - max(open_val, close_val)
        lower_wick = min(open_val, close_val) - low_val
        body_size = abs(close_val - open_val)
        
        # Wick rejection conditions
        long_wick_rejection = (
            lower_wick > 0.6 * daily_range and  # Long lower wick (bullish rejection)
            body_size < 0.3 * daily_range and   # Small body
            upper_wick < 0.2 * daily_range      # Small upper wick
        )
        
        short_wick_rejection = (
            upper_wick > 0.6 * daily_range and  # Long upper wick (bearish rejection)
            body_size < 0.3 * daily_range and   # Small body
            lower_wick < 0.2 * daily_range      # Small lower wick
        )
        
        # Pullback conditions (price near weekly EMA)
        pullback_long = (
            close_val > ema_val and                    # Above weekly EMA (uptrend bias)
            close_val < ema_val + 0.5 * atr_val and    # But not too far above (pullback)
            vol_ratio_val > 1.2                        # Volume confirmation
        )
        
        pullback_short = (
            close_val < ema_val and                    # Below weekly EMA (downtrend bias)
            close_val > ema_val - 0.5 * atr_val and    # But not too far below (pullback)
            vol_ratio_val > 1.2                        # Volume confirmation
        )
        
        if position == 0:
            # Long: Bullish wick rejection + pullback to weekly EMA
            if long_wick_rejection and pullback_long:
                signals[i] = 0.25
                position = 1
            # Short: Bearish wick rejection + pullback to weekly EMA
            elif short_wick_rejection and pullback_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish wick rejection or price breaks below weekly EMA
            if short_wick_rejection or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish wick rejection or price breaks above weekly EMA
            if long_wick_rejection or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals