#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vwap_deviation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate VWAP for each 1d bar (typical price * volume)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Get 1d data for ATR (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = high_1d[0] - close_1d[0]  # first bar
    tr3[0] = high_1d[0] - low_1d[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price position relative to VWAP
    price_vwap_ratio = close / vwap_1d_aligned
    
    # Volatility filter: only trade when volatility is elevated
    vol_filter = atr_1d_aligned > 0  # Ensure ATR is valid
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(price_vwap_ratio[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or not vol_filter[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: price deviates significantly from VWAP
        # Long when price is significantly below VWAP (oversold)
        # Short when price is significantly above VWAP (overbought)
        deviation = price_vwap_ratio[i] - 1.0
        
        long_signal = deviation < -0.02  # 2% below VWAP
        short_signal = deviation > 0.02   # 2% above VWAP
        
        # Exit when price returns to VWAP (mean reversion)
        exit_long = deviation > -0.005  # Within 0.5% of VWAP
        exit_short = deviation < 0.005   # Within 0.5% of VWAP
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals