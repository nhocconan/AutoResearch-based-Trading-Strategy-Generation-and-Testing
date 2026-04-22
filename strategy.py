#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate 40-period EMA on daily close
    close_daily = df_daily['close'].values
    ema_40 = pd.Series(close_daily).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Calculate daily ATR(14)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily EMA and ATR to 12h timeframe
    ema_40_aligned = align_htf_to_ltf(prices, df_daily, ema_40)
    atr_14_aligned = align_htf_to_ltf(prices, df_daily, atr_14)
    
    # Calculate 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema_40_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above daily EMA40, RSI < 30 (oversold), volume confirmation
            if (close[i] > ema_40_aligned[i] and 
                rsi[i] < 30 and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below daily EMA40, RSI > 70 (overbought), volume confirmation
            elif (close[i] < ema_40_aligned[i] and 
                  rsi[i] > 70 and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or opposite extreme
            if position == 1:
                if rsi[i] >= 50:  # Exit long when RSI reaches midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] <= 50:  # Exit short when RSI reaches midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_EMA40_RSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0