# 1h_Liquidity_Rebalance_4hTrend
# Hypothesis: Price tends to revert to the 4h VWAP during low-volatility periods, creating mean-reversion opportunities.
# Uses 4h VWAP as the mean and 1h RSI for entry timing. Works in both bull/bear markets as it fades deviations from the mean.
# Target: 60-150 total trades over 4 years (15-37/year) with strict entry conditions.

#!/usr/bin/env python3
name = "1h_Liquidity_Rebalance_4hTrend"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for VWAP calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h VWAP (Volume Weighted Average Price)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_4h = (typical_price_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_values = vwap_4h.values
    
    # Align 4h VWAP to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_values)
    
    # 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr = np.zeros(n)
    atr[:atr_period] = np.nan
    for i in range(atr_period, n):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(hours[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr[i] > 0.5 * np.nanmedian(atr[max(0, i-50):i+1])
        
        if position == 0:
            # Long: Price below 4h VWAP, RSI oversold, in session, adequate volatility
            if (close[i] < vwap_4h_aligned[i] and 
                rsi[i] < 30 and 
                in_session and 
                vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price above 4h VWAP, RSI overbought, in session, adequate volatility
            elif (close[i] > vwap_4h_aligned[i] and 
                  rsi[i] > 70 and 
                  in_session and 
                  vol_filter):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price crosses above VWAP OR RSI overbought
            if (close[i] >= vwap_4h_aligned[i] or rsi[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price crosses below VWAP OR RSI oversold
            if (close[i] <= vwap_4h_aligned[i] or rsi[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals