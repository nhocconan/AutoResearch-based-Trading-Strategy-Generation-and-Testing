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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2) for mean reversion signals
    close_series = pd.Series(df_1d['close'])
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Calculate 1d RSI(14) for overbought/oversold conditions
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d ATR(14) for volatility filter
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    close_series_1d = pd.Series(df_1d['close'])
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series_1d.shift(1))
    tr3 = abs(low_series - close_series_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band.values)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band.values)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Calculate 6-period average volume for confirmation (6h * 6 = 36h ~ 1.5 days)
    vol_avg = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        # Volume confirmation: current volume > 2x average
        vol_confirm = vol > (vol_avg[i] * 2.0) if not np.isnan(vol_avg[i]) else False
        
        # Mean reversion signals from Bollinger Bands
        price_above_upper = price > upper_band_aligned[i]
        price_below_lower = price < lower_band_aligned[i]
        
        # RSI extreme conditions
        rsi_overbought = rsi_aligned[i] > 70
        rsi_oversold = rsi_aligned[i] < 30
        
        if position == 0:
            # Short setup: price above upper band + RSI overbought + volume confirmation + volatility filter
            if (price_above_upper and rsi_overbought and vol_confirm and vol_filter):
                position = -1
                signals[i] = -position_size
            # Long setup: price below lower band + RSI oversold + volume confirmation + volatility filter
            elif (price_below_lower and rsi_oversold and vol_confirm and vol_filter):
                position = 1
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back above lower band OR RSI returns above 50
            if price > lower_band_aligned[i] or rsi_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back below upper band OR RSI returns below 50
            if price < upper_band_aligned[i] or rsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dBB_RSI_MeanReversion"
timeframe = "6h"
leverage = 1.0