#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# - Long: price breaks above Donchian(20) high, volume > 1.8x 20-period avg, ATR(14) > ATR(50) (strong trend)
# - Short: price breaks below Donchian(20) low, volume > 1.8x 20-period avg, ATR(14) > ATR(50) (strong trend)
# - Exit: price returns to Donchian midpoint
# - Uses 1w EMA(50) for trend bias filter (only trade in direction of weekly trend)
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits for 12h
# - Works in both bull and bear by requiring strong trending conditions (ATR expansion) and volume confirmation
# - Weekly trend filter prevents counter-trend trading in ranging markets

name = "12h_1w_donchian_atr_volume_v1"
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
    
    # Load 1w data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR filters for regime detection
    # ATR(14) for current volatility
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR(50) for longer-term volatility comparison
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        mid_channel = donchian_mid[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average (strict to reduce trades)
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
        # Trend filter: ATR(14) > ATR(50) (indicates strong trending market with expanding volatility)
        atr_trend = atr_14[i] > atr_50[i]
        
        # 1w EMA trend bias
        ema_bias_long = close_price > ema_50_1w_aligned[i]
        ema_bias_short = close_price < ema_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above upper Donchian, volume confirmation, strong trend, long bias
        if close_price > upper_channel and vol_confirm and atr_trend and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below lower Donchian, volume confirmation, strong trend, short bias
        if close_price < lower_channel and vol_confirm and atr_trend and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Donchian midpoint
            exit_long = close_price <= mid_channel
        elif position == -1:
            # Exit short if price returns to Donchian midpoint
            exit_short = close_price >= mid_channel
        
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