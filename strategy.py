#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and chop regime filter
# - Long: price breaks above Donchian(20) high, volume > 1.5x 20-period avg, CHOP(14) < 38.2 (trending)
# - Short: price breaks below Donchian(20) low, volume > 1.5x 20-period avg, CHOP(14) < 38.2 (trending)
# - Exit: price returns to Donchian midpoint or ATR-based stop
# - Uses 12h EMA(50) trend filter: price > EMA for long bias, price < EMA for short bias
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 25-40 trades/year (100-160 total over 4 years) to stay within fee drag limits
# - Donchian channels work well in trending markets; chop filter avoids ranging conditions

name = "4h_12h_donchian_volume_chop_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    dc_mid = (dc_high + dc_low) / 2
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (CHOP) for regime detection
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max(HH) - Min(LL) over 14 periods
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP formula: 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop = np.nan_to_num(chop, nan=50.0)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(dc_mid[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        dc_high_level = dc_high[i]
        dc_low_level = dc_low[i]
        dc_mid_level = dc_mid[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Chop regime filter: CHOP < 38.2 indicates trending market (avoid ranging)
        chop_trend = chop[i] < 38.2
        
        # 12h EMA trend bias
        ema_bias_long = close_price > ema_50_12h_aligned[i]
        ema_bias_short = close_price < ema_50_12h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Donchian high, volume confirmation, trending chop, long bias
        if close_price > dc_high_level and vol_confirm and chop_trend and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Donchian low, volume confirmation, trending chop, short bias
        if close_price < dc_low_level and vol_confirm and chop_trend and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Donchian midpoint
            exit_long = close_price <= dc_mid_level
        elif position == -1:
            # Exit short if price returns to Donchian midpoint
            exit_short = close_price >= dc_mid_level
        
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