#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week EMA trend filter and volume confirmation
# Long when price breaks above 20-day high + close > weekly EMA(20) + volume > 1.5x 20-day average volume
# Short when price breaks below 20-day low + close < weekly EMA(20) + volume > 1.5x 20-day average volume
# Exit when price crosses 10-day EMA (trend reversal) or volatility contraction (ATR ratio < 0.8)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA for trend filter to avoid counter-trend trades
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20)
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 1-day Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit signal
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR(14) for stoploss and volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    volume_s = pd.Series(volume)
    vol_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR ratio for volatility contraction exit (current ATR / 20-day average ATR)
    atr_s = pd.Series(atr)
    atr_ma_20 = atr_s.rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / (atr_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_10[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 10-day EMA or volatility contraction
            elif close[i] < ema_10[i] or atr_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 10-day EMA or volatility contraction
            elif close[i] > ema_10[i] or atr_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend and volume filters
            # Long: break above 20-day high + close > weekly EMA + volume spike
            if close[i] > donchian_high[i] and close[i] > ema_20_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below 20-day low + close < weekly EMA + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_20_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals