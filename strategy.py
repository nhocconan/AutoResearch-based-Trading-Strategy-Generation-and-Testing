#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for direction, 1h for entry timing.
# Uses 4h EMA20 for trend direction, 1d RSI for overbought/oversold conditions, 
# and 1h volume spike for entry timing. Trades only in 08-20 UTC session.
# Designed to work in both bull and bear markets by filtering trend and using mean reversion entries.
# Target: 60-150 total trades over 4 years.

name = "1h_4h_1d_EMA_RSI_Volume_Spike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === 4h: EMA20 for trend direction ===
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d: RSI(14) for overbought/oversold ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.fillna(50).values  # fill NaN with 50 (neutral)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # === 1h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average) with min_periods
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        ema_val = ema_4h_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (price > EMA), RSI not overbought, volume spike
            if close_val > ema_val and rsi_val < 70 and vol_ratio_val > 2.0:
                signals[i] = 0.20
                position = 1
            # Short: Downtrend (price < EMA), RSI not oversold, volume spike
            elif close_val < ema_val and rsi_val > 30 and vol_ratio_val > 2.0:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA or RSI overbought
            if close_val < ema_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price crosses above EMA or RSI oversold
            if close_val > ema_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals