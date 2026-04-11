#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HTF trend filter and volume confirmation
# - Long when price breaks above 4h Donchian upper band with 1d close > 1d EMA(50) and volume > 1.5x 20-period average
# - Short when price breaks below 4h Donchian lower band with 1d close < 1d EMA(50) and volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF trend filter ensures we trade with the higher timeframe direction, reducing false signals
# - Volume confirmation ensures breakouts have conviction

name = "4h_1d_donchian_volume_trend_v3"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_close > donchian_upper[i-1]  # Close above previous period's upper band
        breakout_short = price_close < donchian_lower[i-1]  # Close below previous period's lower band
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Trend filter: 1d close above/below EMA(50)
        trend_long = close[i] > ema_50_1d_aligned[i]  # Current 4h close > 1d EMA(50)
        trend_short = close[i] < ema_50_1d_aligned[i]  # Current 4h close < 1d EMA(50)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian upper breakout + volume confirmation + 1d uptrend
        if breakout_long and vol_confirm and trend_long:
            enter_long = True
        
        # Short: Donchian lower breakdown + volume confirmation + 1d downtrend
        if breakout_short and vol_confirm and trend_short:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below lower band OR trend turns down
            exit_long = (price_close < donchian_lower[i-1]) or (not trend_long)
        elif position == -1:
            # Exit short if price breaks above upper band OR trend turns up
            exit_short = (price_close > donchian_upper[i-1]) or (not trend_short)
        
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