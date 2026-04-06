#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction and 1d RSI extremes for mean reversion bias.
# In bull markets: follow 4h Supertrend long. In bear markets: fade 1d RSI extremes (buy RSI<30, sell RSI>70).
# Uses volume confirmation to avoid false signals. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_supertrend4d_rsi_ext_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on 4h
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Final Bands
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    for i in range(1, len(df_4h)):
        if df_4h['close'].iloc[i] <= final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(final_upper.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]
        if df_4h['close'].iloc[i] >= final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(final_lower.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]
    
    # Supertrend
    supertrend = np.zeros(len(df_4h))
    for i in range(len(df_4h)):
        if i == 0:
            supertrend[i] = final_upper.iloc[i]
        elif supertrend[i-1] == final_upper.iloc[i-1]:
            supertrend[i] = final_lower.iloc[i] if df_4h['close'].iloc[i] <= final_upper.iloc[i-1] else final_upper.iloc[i]
        else:
            supertrend[i] = final_upper.iloc[i] if df_4h['close'].iloc[i] >= final_lower.iloc[i-1] else final_lower.iloc[i]
    
    # Align Supertrend to 1h (trend direction: 1 for uptrend, -1 for downtrend)
    trend_4h = np.where(supertrend == final_upper.values, -1, 1)  # -1: downtrend, 1: uptrend
    trend_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on 1d
    delta = df_1d['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(trend_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session and volume filters
        hour = hours[i]
        in_session = 8 <= hour <= 20
        vol_filter = volume[i] > vol_ma[i]
        
        if not (in_session and vol_filter):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: 4h trend turns bearish OR 1d RSI overbought
            if trend_aligned[i] == -1 or rsi_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: 4h trend turns bullish OR 1d RSI oversold
            if trend_aligned[i] == 1 or rsi_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with filters
            if in_session and vol_filter:
                # Follow 4h trend in bull/bear, but fade RSI extremes
                if trend_aligned[i] == 1 and rsi_aligned[i] < 30:  # uptrend + oversold = long
                    signals[i] = 0.20
                    position = 1
                elif trend_aligned[i] == -1 and rsi_aligned[i] > 70:  # downtrend + overbought = short
                    signals[i] = -0.20
                    position = -1
                # Counter-trend only at extreme RSI (mean reversion in ranging markets)
                elif trend_aligned[i] == -1 and rsi_aligned[i] < 30:  # downtrend but deeply oversold = long
                    signals[i] = 0.20
                    position = 1
                elif trend_aligned[i] == 1 and rsi_aligned[i] > 70:  # uptrend but deeply overbought = short
                    signals[i] = -0.20
                    position = -1
    
    return signals