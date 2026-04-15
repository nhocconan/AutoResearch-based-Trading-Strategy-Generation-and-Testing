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
    
    # Get daily data for 1d HTF context
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Calculate daily EMA(21) for trend direction
    daily_ema_21 = pd.Series(daily['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    daily_ema_21_aligned = align_htf_to_ltf(prices, daily, daily_ema_21)
    
    # Calculate daily EMA(55) for trend confirmation
    daily_ema_55 = pd.Series(daily['close'].values).ewm(span=55, adjust=False, min_periods=55).mean().values
    daily_ema_55_aligned = align_htf_to_ltf(prices, daily, daily_ema_55)
    
    # Calculate daily volume average for volume spike detection
    vol_ma_10d = pd.Series(daily['volume'].values).rolling(window=10, min_periods=10).mean().values
    vol_ma_10d_aligned = align_htf_to_ltf(prices, daily, vol_ma_10d)
    
    # Volume filter: current volume > 1.5x 10-day average volume
    vol_threshold = 1.5 * vol_ma_10d_aligned
    vol_spike = volume > vol_threshold
    
    # Calculate daily RSI(14) for momentum filter
    delta = pd.Series(daily['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14d = 100 - (100 / (1 + rs))
    rsi_14d_values = rsi_14d.values
    rsi_14d_aligned = align_htf_to_ltf(prices, daily, rsi_14d_values)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(daily_ema_21_aligned[i]) or 
            np.isnan(daily_ema_55_aligned[i]) or np.isnan(vol_ma_10d_aligned[i]) or 
            np.isnan(rsi_14d_aligned[i])):
            continue
        
        # Volatility filter: avoid low volatility chop
        if atr_14d_aligned[i] < (0.005 * close[i]):
            signals[i] = 0.0
            continue
            
        # Long: Price above EMA21 and EMA55 + volume spike + RSI > 50
        if (close[i] > daily_ema_21_aligned[i] and 
            close[i] > daily_ema_55_aligned[i] and 
            vol_spike[i] and 
            rsi_14d_aligned[i] > 50):
            signals[i] = 0.25
        
        # Short: Price below EMA21 and EMA55 + volume spike + RSI < 50
        elif (close[i] < daily_ema_21_aligned[i] and 
              close[i] < daily_ema_55_aligned[i] and 
              vol_spike[i] and 
              rsi_14d_aligned[i] < 50):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < daily_ema_21_aligned[i] and signals[i-1] > 0) or \
             (close[i] > daily_ema_21_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_DailyEMA21_55_Volume_RSI_Filter"
timeframe = "12h"
leverage = 1.0