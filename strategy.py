#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_RSI2_Cci_Trend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h RSI(2) + CCI(14) with 1d EMA(34) trend filter.
    - Long: RSI(2) < 15 AND CCI(14) < -100 AND close > 1d EMA(34)
    - Short: RSI(2) > 85 AND CCI(14) > 100 AND close < 1d EMA(34)
    - Exit: RSI(2) crosses 50 or price crosses opposite side of 1d EMA(34)
    - Uses extreme RSI(2) for mean reversion in trending markets
    - Target: 12-25 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate RSI(2)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi2 = 100 - (100 / (1 + rs))
    rsi2 = rsi2.values
    
    # Calculate CCI(14)
    typical_price = (high + low + close) / 3
    tp_s = pd.Series(typical_price)
    sma_tp = tp_s.rolling(window=14, min_periods=14).mean()
    mad = tp_s.rolling(window=14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma_tp.values) / (0.015 * mad.values)
    cci = np.nan_to_num(cci, nan=0.0)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Extreme oversold + CCI sell signal + above trend
            if rsi2[i] < 15 and cci[i] < -100 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Extreme overbought + CCI buy signal + below trend
            elif rsi2[i] > 85 and cci[i] > 100 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses 50 or price breaks below trend
            if rsi2[i] > 50 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses 50 or price breaks above trend
            if rsi2[i] < 50 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals