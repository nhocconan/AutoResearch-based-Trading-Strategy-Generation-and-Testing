#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_volatility_breakout_v1"
timeframe = "1d"
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
    
    # 1d indicators
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # EMA50 on 1w
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Wait for EMA50
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(vol_filter[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI < 40 or volatility too low
            if rsi[i] < 40 or atr[i] < np.mean(atr[max(0, i-20):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI > 60 or volatility too low
            if rsi[i] > 60 or atr[i] < np.mean(atr[max(0, i-20):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs weekly EMA50
            bullish_trend = close[i] > ema_50_1w_aligned[i]
            bearish_trend = close[i] < ema_50_1w_aligned[i]
            
            # Volatility filter: current ATR > average of last 20 periods
            vol_filter_now = atr[i] > np.mean(atr[max(0, i-20):i]) * 1.2 if i >= 20 else False
            
            # Long: RSI < 30 (oversold) + bullish trend + volatility expansion + volume
            if (rsi[i] < 30 and 
                bullish_trend and 
                vol_filter_now and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: RSI > 70 (overbought) + bearish trend + volatility expansion + volume
            elif (rsi[i] > 70 and 
                  bearish_trend and 
                  vol_filter_now and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals