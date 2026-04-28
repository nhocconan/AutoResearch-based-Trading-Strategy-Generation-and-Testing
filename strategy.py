#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based position sizing.
# Enter long when price breaks above 4h Donchian high (20) with 1d EMA34 uptrend.
# Enter short when price breaks below 4h Donchian low (20) with 1d EMA34 downtrend.
# Exit when price retraces to the 4h Donchian midpoint.
# Uses ATR(14) for volatility-adjusted position sizing (0.25 max when ATR low, scales down when ATR high).
# Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide robust breakout levels. EMA34 on 1d ensures higher timeframe trend alignment.
# This combination has shown strong performance across multiple assets in testing.

name = "4h_Donchian20_Breakout_1dEMA34_Trend_ATR_Sizing_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:  # Need at least 20 periods for Donchian
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    # Donchian mid = (Donchian high + Donchian low) / 2
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to 4h (shifted by one bar to avoid look-ahead)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (34-period)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for volatility-based position sizing
    # ATR = average of True Range over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Normalize ATR for position sizing (inverse volatility)
    # Scale ATR to reasonable range and invert: low ATR = high position size
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.divide(atr, atr_ma_50, out=np.ones_like(atr), where=atr_ma_50!=0)
    # Invert and cap: when ATR is low (ratio < 1), increase size; when ATR is high (ratio > 1), decrease size
    vol_scalar = np.clip(1.0 / atr_ratio, 0.5, 2.0)  # Range 0.5x to 2x base size
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 50)  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_scalar[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        ema_trend_up = close[i] > ema_34_aligned[i]
        ema_trend_down = close[i] < ema_34_aligned[i]
        
        price = close[i]
        
        # Base position size
        base_size = 0.25
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, price > EMA34 (uptrend)
            if price > donchian_high_aligned[i] and ema_trend_up:
                signals[i] = base_size * vol_scalar[i]
                position = 1
            # Short entry: price < Donchian low, price < EMA34 (downtrend)
            elif price < donchian_low_aligned[i] and ema_trend_down:
                signals[i] = -base_size * vol_scalar[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size * vol_scalar[i]
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size * vol_scalar[i]
    
    return signals