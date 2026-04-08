#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h_trend_volume_v1"
timeframe = "1h"
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
    
    # 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily timeframe
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.values
    
    # Align RSI to 1h timeframe
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # ATR for volatility filter (14-period on 1h)
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: ATR > 20-period ATR mean (avoid choppy markets)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma
    
    # Volume filter: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend reverses
            if rsi_14_1d_aligned[i] > 70 or close[i] < ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend reverses
            if rsi_14_1d_aligned[i] < 30 or close[i] > ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend filter: price vs 4h EMA21
            uptrend = close[i] > ema_21_4h_aligned[i]
            downtrend = close[i] < ema_21_4h_aligned[i]
            
            # Long: RSI < 30 (oversold pullback) + uptrend + volume spike + vol filter
            if (rsi_14_1d_aligned[i] < 30 and 
                uptrend and 
                vol_spike[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short: RSI > 70 (overbought pullback) + downtrend + volume spike + vol filter
            elif (rsi_14_1d_aligned[i] > 70 and 
                  downtrend and 
                  vol_spike[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals