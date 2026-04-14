#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze with volume confirmation and 1-day ADX trend filter
# Long when Bollinger Bands width < 20th percentile AND price > upper band AND volume > 1.5x average AND ADX > 25
# Short when Bollinger Bands width < 20th percentile AND price < lower band AND volume > 1.5x average AND ADX > 25
# Exit when price crosses the middle Bollinger Band (SMA20)
# Bollinger squeeze identifies low volatility periods preceding breakouts
# Volume confirmation ensures institutional participation
# ADX filter ensures we only trade in trending markets to avoid whipsaws in ranging conditions
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing explosive moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    bollinger_width = upper - lower
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    width_series = pd.Series(bollinger_width)
    width_percentile = width_series.rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate Average True Range for ADX
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Calculate Directional Movement for ADX
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate smoothed DM and ATR for ADX
    atr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align indicators to 4h timeframe
    width_percentile_aligned = align_htf_to_ltf(prices, df_1d, width_percentile)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper.values)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower.values)
    basis_aligned = align_htf_to_ltf(prices, df_1d, basis.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(width_percentile_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(basis_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        volume_current = volume[i]
        
        if position == 0:
            # Bollinger squeeze condition: width < 20th percentile
            squeeze_condition = width_percentile_aligned[i] < 20
            
            # Volume confirmation: current volume > 1.5x average
            volume_condition = volume_current > 1.5 * avg_volume_aligned[i]
            
            # Trend filter: ADX > 25
            trend_condition = adx_aligned[i] > 25
            
            # Long setup: squeeze + volume + trend + price > upper band
            if (squeeze_condition and volume_condition and trend_condition and
                price > upper_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: squeeze + volume + trend + price < lower band
            elif (squeeze_condition and volume_condition and trend_condition and
                  price < lower_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle band (SMA20)
            if price < basis_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above middle band (SMA20)
            if price > basis_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_BollingerSqueeze_Volume_ADX"
timeframe = "4h"
leverage = 1.0