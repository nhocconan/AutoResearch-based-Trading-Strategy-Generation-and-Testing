#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based stoploss.
# Enter long when price breaks above Donchian upper band with 1d EMA34 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below Donchian lower band with 1d EMA34 downtrend and volume > 1.5x 20-bar average.
# Exit long when price retraces to Donchian middle band (20-bar average of high/low) or ATR stoploss hit.
# Exit short when price retraces to Donchian middle band or ATR stoploss hit.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure; 1d EMA34 ensures higher timeframe alignment; volume spike filters weak breakouts.
# ATR stoploss manages risk in both bull and bear markets.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian(20) - 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2  # Middle band for exit
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR stoploss
    
    start_idx = 20  # Ensure sufficient history for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle exits first (stoploss or middle band retracement)
        if position == 1:  # Long position
            # Stoploss: price <= entry_price - 2.0 * atr[i]
            # Middle band exit: price <= donchian_mid[i]
            if price <= entry_price - 2.0 * atr[i] or price <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Stoploss: price >= entry_price + 2.0 * atr[i]
            # Middle band exit: price >= donchian_mid[i]
            if price >= entry_price + 2.0 * atr[i] or price >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:  # Flat - look for new entries
            # Long entry: price > donchian_high[i], EMA34 up, volume confirm
            if price > donchian_high[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price  # Record entry price for stoploss
            # Short entry: price < donchian_low[i], EMA34 down, volume confirm
            elif price < donchian_low[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price  # Record entry price for stoploss
            else:
                signals[i] = 0.0
    
    return signals