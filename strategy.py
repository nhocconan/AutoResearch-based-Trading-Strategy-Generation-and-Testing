#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 12-hour EMA(50) filter and volume confirmation
# Long when Williams %R crosses above -20 from oversold, price > 12h EMA(50), volume > 1.5x 6h EMA(20) volume
# Short when Williams %R crosses below -80 from overbought, price < 12h EMA(50), volume > 1.5x 6h EMA(20) volume
# Exit when Williams %R returns to -50 or opposite crossover occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_williamsr_12h_ema50_vol_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h Williams %R(14)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_6h) / (highest_high - lowest_low)) * -100,
        -50
    )
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # 6h EMA(20) volume for confirmation
    volume_6h = df_6h['volume'].values
    volume_ema_6h = pd.Series(volume_6h).ewm(span=20, adjust=False).mean().values
    volume_ema_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ema_6h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_ema_6h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns to -50 or crosses below -80
            elif williams_r_aligned[i] >= -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns to -50 or crosses above -20
            elif williams_r_aligned[i] <= -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Williams %R crossover signals
            williams_prev = williams_r_aligned[i-1] if i > 0 else -50
            williams_curr = williams_r_aligned[i]
            
            # Long: Williams %R crosses above -20 from oversold
            if (williams_prev <= -20 and williams_curr > -20 and
                close[i] > ema_12h_aligned[i] and
                volume[i] > 1.5 * volume_ema_6h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R crosses below -80 from overbought
            elif (williams_prev >= -80 and williams_curr < -80 and
                  close[i] < ema_12h_aligned[i] and
                  volume[i] > 1.5 * volume_ema_6h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals