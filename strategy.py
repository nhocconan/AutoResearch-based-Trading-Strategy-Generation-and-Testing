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
    
    # Load 12h data (HTF) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-period RSI on 12h closes
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=12, adjust=False, min_periods=12).mean().values
    avg_loss = pd.Series(loss).ewm(span=12, adjust=False, min_periods=12).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Calculate 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_shift = np.roll(close_12h, 1)
    close_12h_shift[0] = close_12h[0]
    tr = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - close_12h_shift), np.abs(low_12h - close_12h_shift)))
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume average (20-period)
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Calculate 4h Bollinger Bands (20, 2.0)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + 2 * std_20
    lower_band = ma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_avg_20_12h_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches lower Bollinger Band, RSI oversold (<30), and volume spike
            if (close[i] <= lower_band[i] and 
                rsi_12h_aligned[i] < 30 and 
                volume[i] > 2.0 * vol_avg_20_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band, RSI overbought (>70), and volume spike
            elif (close[i] >= upper_band[i] and 
                  rsi_12h_aligned[i] > 70 and 
                  volume[i] > 2.0 * vol_avg_20_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to middle (mean reversion complete)
            if position == 1:
                if close[i] >= ma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= ma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_BollingerBand_RSI12h_VolumeSpike_MeanReversion"
timeframe = "4h"
leverage = 1.0