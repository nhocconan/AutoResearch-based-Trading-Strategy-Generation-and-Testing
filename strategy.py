#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1d RSI extremes for mean reversion entries.
# In bull markets: 4h uptrend + 1d RSI < 30 → long (buy dips in uptrend)
# In bear markets: 4h downtrend + 1d RSI > 70 → short (sell rallies in downtrend)
# Volume confirmation (>1.5x 20-bar MA) ensures institutional participation.
# Session filter (08-20 UTC) reduces noise outside active trading hours.
# Discrete sizing (0.20) minimizes fee churn. Target: 60-150 trades over 4 years.

name = "1h_Supertrend4h_RSI1dExtremes_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # 4h Supertrend (ATR=10, mult=3.0)
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    atr_10 = pd.Series(high_4h := df_4h['high'] - df_4h['low']).rolling(window=10, min_periods=10).mean()
    tr = np.maximum(high_4h, np.maximum(df_4h['high'].shift(1), df_4h['low'].shift(1))) - np.minimum(low_4h, np.minimum(df_4h['high'].shift(1), df_4h['low'].shift(1)))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean()
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # Initialize Supertrend
    supertrend = np.full(len(df_4h), np.nan)
    direction = np.full(len(df_4h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(df_4h)):
        if i == 10:
            supertrend[i] = lower_band.iloc[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upper_band.iloc[i-1]:
                supertrend[i] = lower_band.iloc[i] if df_4h['close'].iloc[i] <= upper_band.iloc[i] else upper_band.iloc[i]
                direction[i] = -1 if df_4h['close'].iloc[i] <= upper_band.iloc[i] else 1
            else:
                supertrend[i] = upper_band.iloc[i] if df_4h['close'].iloc[i] >= lower_band.iloc[i] else lower_band.iloc[i]
                direction[i] = 1 if df_4h['close'].iloc[i] >= lower_band.iloc[i] else -1
    
    # Align 4h Supertrend and direction to 1h
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1d HTF data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d RSI(14)
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align 1d RSI to 1h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h uptrend (direction=1) + 1d RSI < 30 (oversold) + volume spike
            if direction_aligned[i] == 1 and rsi_aligned[i] < 30 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (direction=-1) + 1d RSI > 70 (overbought) + volume spike
            elif direction_aligned[i] == -1 and rsi_aligned[i] > 70 and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 4h downtrend or 1d RSI > 70 (overbought)
            if direction_aligned[i] == -1 or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on 4h uptrend or 1d RSI < 30 (oversold)
            if direction_aligned[i] == 1 or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals