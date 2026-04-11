#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR filter
# - Donchian breakout on 12h: price > 20-period high for long, price < 20-period low for short
# - Volume confirmation: 12h volume > 1.5x 20-period average volume
# - ATR filter: ATR(14) > 0.5 * ATR(50) to ensure sufficient volatility
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - Volume confirmation filters out weak breakouts
# - ATR filter avoids ranging markets where breakouts fail
# - 1d HTF ensures we use completed daily candles for volume and ATR calculations

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume and ATR confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d ATR (14-period) and ATR (50-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volatility filter: ATR(14) > 0.5 * ATR(50)
    vol_filter = atr_14_aligned > 0.5 * atr_50_aligned
    
    # Pre-compute 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_close > donchian_high[i-1]  # Close above previous period's high
        breakout_short = price_close < donchian_low[i-1]  # Close below previous period's low
        
        # Volume confirmation: current 12h volume > 1.5x 1d volume SMA
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ATR volatility filter
        vol_ok = vol_filter[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout + volume confirmation + volatility filter
        if breakout_long and vol_confirm and vol_ok:
            enter_long = True
        
        # Short: Donchian breakdown + volume confirmation + volatility filter
        if breakout_short and vol_confirm and vol_ok:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Donchian breakdown
            exit_long = price_close < donchian_low[i-1]
        elif position == -1:
            # Exit short if Donchian breakout
            exit_short = price_close > donchian_high[i-1]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals