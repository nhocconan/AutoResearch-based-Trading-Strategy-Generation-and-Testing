#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and ATR-based volatility filter
# - Long when price breaks above Donchian(20) high AND 12h EMA(21) > EMA(50) AND ATR(14) < 1.5 * ATR(50) (low volatility regime)
# - Short when price breaks below Donchian(20) low AND 12h EMA(21) < EMA(50) AND ATR(14) < 1.5 * ATR(50)
# - Exit when price crosses the Donchian(20) midline (10-period average of high/low)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian breakouts capture momentum; 12h EMA filter ensures alignment with higher timeframe trend
# - ATR volatility filter avoids high-volatility choppy markets where breakouts fail
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: trend filter prevents counter-trend trades, volatility filter adapts to regime

name = "4h_12h_donchian_ema_atr_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA trend filter: EMA(21) vs EMA(50)
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish_12h = ema_21_12h > ema_50_12h
    ema_bearish_12h = ema_21_12h < ema_50_12h
    
    # Pre-compute ATR volatility filter: ATR(14) < 1.5 * ATR(50) (low volatility regime)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = np.abs(high[0] - close[0])  # First bar
    tr3[0] = np.abs(low[0] - close[0])  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, min_periods=50, adjust=False).mean().values
    low_volatility = atr_14 < (1.5 * atr_50)
    
    # Pre-compute Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align HTF indicators to 4h timeframe
    ema_bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bullish_12h)
    ema_bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bearish_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_12h_aligned[i]) or np.isnan(ema_bearish_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(low_volatility[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 12h bullish trend AND low volatility
            if (close[i] > donchian_high[i] and 
                ema_bullish_12h_aligned[i] and 
                low_volatility[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 12h bearish trend AND low volatility
            elif (close[i] < donchian_low[i] and 
                  ema_bearish_12h_aligned[i] and 
                  low_volatility[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Donchian midline
            # Exit when price crosses the Donchian midline
            exit_signal = (position == 1 and close[i] < donchian_mid[i]) or \
                          (position == -1 and close[i] > donchian_mid[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals