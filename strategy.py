#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HTF trend filter and volume confirmation
# - Uses 1d EMA(50) as trend filter: long only when price > EMA50, short only when price < EMA50
# - Donchian breakout from 4h provides entry timing with volume confirmation (>1.5x 20-bar avg volume)
# - ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14)
# - Discrete position sizing: ±0.25 to balance return and drawdown
# - Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag
# - Trend filter reduces false breakouts in choppy markets, works in both bull and bear regimes

name = "4h_1d_donchian_trend_volume_v1"
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
    entry_price = 0.0
    atr_value = 0.0
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h ATR(14) for stoploss
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 4h Donchian channels (20-period)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_close > donchian_upper[i-1]  # Close above previous period's upper band
        breakout_short = price_close < donchian_lower[i-1]  # Close below previous period's lower band
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        trend_long = price_close > ema_50_1d_aligned[i]
        trend_short = price_close < ema_50_1d_aligned[i]
        
        # Update ATR for stoploss calculation
        atr_value = atr_14[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian upper breakout + volume confirmation + uptrend filter
        if breakout_long and vol_confirm and trend_long:
            enter_long = True
        
        # Short: Donchian lower breakdown + volume confirmation + downtrend filter
        if breakout_short and vol_confirm and trend_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: stoploss hit OR Donchian lower break
            if entry_price > 0:
                stoploss_level = entry_price - 2.5 * atr_value
                exit_long = (price_close <= stoploss_level) or (price_close < donchian_lower[i-1])
            else:
                exit_long = price_close < donchian_lower[i-1]
        elif position == -1:
            # Exit short: stoploss hit OR Donchian upper break
            if entry_price > 0:
                stoploss_level = entry_price + 2.5 * atr_value
                exit_short = (price_close >= stoploss_level) or (price_close > donchian_upper[i-1])
            else:
                exit_short = price_close > donchian_upper[i-1]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals