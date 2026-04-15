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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr_daily).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA(34) for trend direction
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d RSI(14) for momentum
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 4h timeframe
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    atr_daily_4h = align_htf_to_ltf(prices, df_1d, atr_daily)
    
    # Calculate 4h ATR(14) for stoploss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(atr_daily_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price > 1d EMA34 (uptrend) + RSI > 50 (bullish momentum) → long
        # 2. 4h price < 1d EMA34 (downtrend) + RSI < 50 (bearish momentum) → short
        # 3. Volatility filter: 4h ATR > 0.3% of price (avoid extremely low volatility)
        # 4. Volume confirmation: volume > 1.2x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 4h price above daily EMA34 with bullish momentum
        if (close[i] > ema_34_4h[i] and            # 4h price above daily EMA34
            rsi_4h[i] > 50 and                     # Bullish momentum
            volume_ratio[i] > 1.2 and              # Volume confirmation
            atr_14_4h[i] > 0.003 * close[i]):      # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 4h price below daily EMA34 with bearish momentum
        elif (close[i] < ema_34_4h[i] and          # 4h price below daily EMA34
              rsi_4h[i] < 50 and                   # Bearish momentum
              volume_ratio[i] > 1.2 and            # Volume confirmation
              atr_14_4h[i] > 0.003 * close[i]):    # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA34_RSI_Volume_Volatility_Filter"
timeframe = "4h"
leverage = 1.0