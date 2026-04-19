#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and daily volume confirmation.
# Long when RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) AND volume > 1.3x daily average volume
# Short when RSI > 70 (overbought) AND price < 4h EMA50 (downtrend) AND volume > 1.3x daily average volume
# Exit when RSI crosses back to neutral (40 for long, 60 for short)
# Uses RSI for mean reversion entries, 4h EMA for trend filter, volume for confirmation.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).
name = "1h_RSI_MeanReversion_TrendFilter_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain_ma / loss_ma
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Get 4h EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d average volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema50 = ema50_4h_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: RSI oversold + price above 4h EMA50 + volume spike
            if rsi_val < 30 and price > ema50 and vol > 1.3 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI overbought + price below 4h EMA50 + volume spike
            elif rsi_val > 70 and price < ema50 and vol > 1.3 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses back to neutral (40)
            if rsi_val > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI crosses back to neutral (60)
            if rsi_val < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals