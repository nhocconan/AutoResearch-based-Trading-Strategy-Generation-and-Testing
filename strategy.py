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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR (14-period) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d RSI(14) for momentum filter
    delta_1d = pd.Series(df_1d['close']).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = (100 - (100 / (1 + rs_1d))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d volume average (20-period) for volume filter
    vol_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        price = close[i]
        vol = volume[i]
        
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5% of price
        vol_filter = atr[i] > (price * 0.005)
        
        # Volume filter: current volume > 1.5x 1d average volume
        vol_confirm = vol > (vol_1d_aligned[i] * 1.5)
        
        # Trend filter: price vs 1d EMA50
        trend_long = price > ema_50_1d_aligned[i]
        trend_short = price < ema_50_1d_aligned[i]
        
        # Momentum filter: RSI between 40 and 60 to avoid extremes
        rsi_ok = (rsi_1d_aligned[i] >= 40) and (rsi_1d_aligned[i] <= 60)
        
        if position == 0:
            # Long: price above EMA50 + volume + volatility + RSI filter
            if trend_long and vol_confirm and vol_filter and rsi_ok:
                position = 1
                signals[i] = position_size
            # Short: price below EMA50 + volume + volatility + RSI filter
            elif trend_short and vol_confirm and vol_filter and rsi_ok:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50
            if price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA50
            if price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dEMA50_Volume_RSI_Filter"
timeframe = "4h"
leverage = 1.0