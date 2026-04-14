#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_series = pd.Series(tr)
    atr_14_1d = tr_series.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate average daily volume (20-period)
    vol_series_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 6-period RSI on 6h data
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.rolling(window=6, min_periods=6).mean().values
    avg_loss = loss_series.rolling(window=6, min_periods=6).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for 20-period volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_1d_aligned[i]
        avg_vol = avg_vol_1d_aligned[i]
        vol = volume[i]
        rsi_val = rsi[i]
        
        # Dynamic thresholds based on volatility
        vol_threshold = 2.0 * avg_vol  # Volume > 2x average
        atr_stop = 1.5 * atr  # Stop distance
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volume spike + price near support
            if (rsi_val < 30 and vol > vol_threshold and 
                price > low[i] + 0.1 * (high[i] - low[i])):  # Not at absolute low
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) + volume spike + price near resistance
            elif (rsi_val > 70 and vol > vol_threshold and 
                  price < high[i] - 0.1 * (high[i] - low[i])):  # Not at absolute high
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or stop loss
            if rsi_val > 50 or price < (entry_price := entry_price if 'entry_price' in locals() else price) - atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 or stop loss
            if rsi_val < 50 or price > (entry_price := entry_price if 'entry_price' in locals() else price) + atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_RSI_Volatility_Mean_Reversion"
timeframe = "6h"
leverage = 1.0