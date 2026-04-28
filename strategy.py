#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for higher timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Exponential Moving Average (34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d Average True Range (14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Relative Strength Index (14) for overbought/oversold conditions
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    
    # Align 1d indicators to 15m timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_1d_aligned[i] > 0.005 * close[i]  # ATR > 0.5% of price
        
        # RSI conditions: avoid extreme overbought/oversold
        rsi_not_overbought = rsi_14_1d_aligned[i] < 75
        rsi_not_oversold = rsi_14_1d_aligned[i] > 25
        
        # Entry conditions
        # Long: uptrend + volume surge + volatility filter + RSI not overbought
        long_entry = trend_up and volume_surge[i] and vol_filter and rsi_not_overbought
        # Short: downtrend + volume surge + volatility filter + RSI not oversold
        short_entry = trend_down and volume_surge[i] and vol_filter and rsi_not_oversold
        
        # Exit conditions: trend reversal or RSI extreme
        long_exit = not trend_up or rsi_14_1d_aligned[i] >= 80
        short_exit = not trend_down or rsi_14_1d_aligned[i] <= 20
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "15m_EMA34_RSI14_Volume_Filter"
timeframe = "15m"
leverage = 1.0