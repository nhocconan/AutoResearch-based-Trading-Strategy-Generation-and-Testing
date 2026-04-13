#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Average Volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 6h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d Bollinger Bands (20, 2) - middle and std
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align 1d Bollinger Bands to 6h
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # 1d RSI (14-period) for trend filter
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    gain_series_1d = pd.Series(gain_1d)
    loss_series_1d = pd.Series(loss_1d)
    avg_gain_1d = gain_series_1d.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss_1d = loss_series_1d.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 14)
    for i in range(start, n):
        if (np.isnan(avg_vol[i]) or np.isnan(rsi[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price below lower BB (oversold) + RSI < 30 + volume spike + 1d RSI < 50 (bearish bias for mean reversion)
            if (price < lower_bb_aligned[i] and rsi[i] < 30 and 
                vol > 1.5 * avg_vol[i] and rsi_1d_aligned[i] < 50):
                position = 1
                signals[i] = position_size
            # Short: price above upper BB (overbought) + RSI > 70 + volume spike + 1d RSI > 50 (bullish bias for mean reversion)
            elif (price > upper_bb_aligned[i] and rsi[i] > 70 and 
                  vol > 1.5 * avg_vol[i] and rsi_1d_aligned[i] > 50):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above SMA (mean reversion complete) OR RSI > 50
            if (price > sma_20_aligned[i] or rsi[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below SMA (mean reversion complete) OR RSI < 50
            if (price < sma_20_aligned[i] or rsi[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Bollinger_RSI_MeanReversion"
timeframe = "6h"
leverage = 1.0