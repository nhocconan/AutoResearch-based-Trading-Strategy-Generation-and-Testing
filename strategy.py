#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour mean reversion with 4h trend and 1d volatility filter.
# Long when: RSI(14) < 30 AND 4h price > 4h VWAP AND 1d ATR(14) < 1d ATR(50) (low vol)
# Short when: RSI(14) > 70 AND 4h price < 4h VWAP AND 1d ATR(14) < 1d ATR(50) (low vol)
# Exit when RSI crosses back to 50.
# Designed for 1h timeframe with low trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 4h VWAP for trend alignment and 1d volatility regime to avoid choppy markets.
# Works in bull markets via mean reversion in uptrend (RSI < 30 + price > VWAP),
# and in bear markets via mean reversion in downtrend (RSI > 70 + price < VWAP).
# Volatility filter (ATR14 < ATR50) avoids high-noise periods and whipsaws.
name = "1h_RSI_4hVWAP_1dATR_VolRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h VWAP for trend alignment
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_num = np.cumsum(typical_price_4h * volume_4h)
    vwap_den = np.cumsum(volume_4h)
    vwap_4h = vwap_num / (vwap_den + 1e-10)
    
    price_above_vwap = close_4h > vwap_4h
    price_below_vwap = close_4h < vwap_4h
    
    price_above_vwap_aligned = align_htf_to_ltf(prices, df_4h, price_above_vwap)
    price_below_vwap_aligned = align_htf_to_ltf(prices, df_4h, price_below_vwap)
    
    # 1d ATR(14) and ATR(50) for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_50  # low volatility regime
    
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(price_above_vwap_aligned[i]) or np.isnan(price_below_vwap_aligned[i]) or 
            np.isnan(low_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 AND 4h price > VWAP AND low vol regime
            long_condition = (rsi[i] < 30) and price_above_vwap_aligned[i] and low_vol_aligned[i]
            # Short: RSI > 70 AND 4h price < VWAP AND low vol regime
            short_condition = (rsi[i] > 70) and price_below_vwap_aligned[i] and low_vol_aligned[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI > 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI < 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals