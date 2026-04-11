#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR (14-period) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_14_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate daily ATR (14-period) for Camarilla pivot calculation
    tr1_d = np.abs(high[1:] - low[1:])
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d = np.concatenate([[np.nan], tr_d])
    atr_14_d = pd.Series(tr_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 20-day average
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required data is invalid
        if (np.isnan(atr_14_1w_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(atr_14_d[i-1])):  # Need previous day's data for pivot
            signals[i] = 0.0
            continue
        
        # Previous day's OHLC for Camarilla calculation
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Calculate Camarilla levels for today
        # Resistance levels
        r4 = pc + ((ph - pl) * 1.1 / 2)
        r3 = pc + ((ph - pl) * 1.1 / 4)
        r2 = pc + ((ph - pl) * 1.1 / 6)
        r1 = pc + ((ph - pl) * 1.1 / 12)
        # Support levels
        s1 = pc - ((ph - pl) * 1.1 / 12)
        s2 = pc - ((ph - pl) * 1.1 / 6)
        s3 = pc - ((ph - pl) * 1.1 / 4)
        s4 = pc - ((ph - pl) * 1.1 / 2)
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Higher timeframe volatility filter: only trade when weekly ATR is elevated
        if i >= 20:  # Need enough history for ATR average
            atr_ma_20 = np.nanmean(atr_14_1w_aligned[max(0, i-19):i+1])
            vol_filter = atr_14_1w_aligned[i] > 0.7 * atr_ma_20
        else:
            vol_filter = True
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above S3 with volume confirmation and volatility filter
        if price_close > s3 and vol_confirm and vol_filter:
            enter_long = True
        
        # Short: Price breaks below R3 with volume confirmation and volatility filter
        if price_close < r3 and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: price returns to the pivot point (PC)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to or below pivot
            exit_long = price_close <= pc
        elif position == -1:
            # Exit short if price returns to or above pivot
            exit_short = price_close >= pc
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals