#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and volatility
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP (typical price * volume cumulative)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # Daily ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA200 for trend filter
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all daily data to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA200
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Volatility filter: only trade when volatility is low (ATR < 0.5 * 20-period avg)
        atr_ma = pd.Series(atr_aligned[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1] if i >= start_idx + 20 else atr_aligned[i]
        low_vol = atr_aligned[i] < 0.5 * atr_ma
        
        # VWAP deviation: look for mean reversion when price deviates significantly from VWAP
        vwap_dev = (close[i] - vwap_aligned[i]) / vwap_aligned[i]
        oversold = vwap_dev < -0.02  # 2% below VWAP
        overbought = vwap_dev > 0.02   # 2% above VWAP
        
        if position == 0:
            # Long: uptrend + low volatility + oversold
            if uptrend and low_vol and oversold:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + low volatility + overbought
            elif downtrend and low_vol and overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or mean reversion back to VWAP
            if not uptrend or vwap_dev > -0.005:  # nearly back to VWAP
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or mean reversion back to VWAP
            if not downtrend or vwap_dev < 0.005:  # nearly back to VWAP
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VWAP_MeanReversion_VolatilityFilter"
timeframe = "12h"
leverage = 1.0