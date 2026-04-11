#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA trend filter
# - Long: price breaks above 20-day Donchian high with volume > 1.5x 20-day avg and price > 1w EMA50
# - Short: price breaks below 20-day Donchian low with volume > 1.5x 20-day avg and price < 1w EMA50
# - Exit: price returns to opposite Donchian level (mean reversion at channel)
# - Uses 1d Donchian channels calculated from prior 20 days, aligned to 1d
# - Uses 1w EMA50 for trend filter (avoid counter-trend trades)
# - Volume confirmation reduces false breakouts
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "1d_1w_donchian_breakout_trend_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Donchian channels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: max(high, lookback=20)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: min(low, lookback=20)
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (use completed 1d bar's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (use completed 1w bar's EMA)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Donchian levels
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # 1w EMA50 trend filter
        ema50 = ema_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper Donchian with volume and uptrend
        if close_price > upper and vol_confirm and close_price > ema50:
            enter_long = True
        
        # Short breakout: price breaks below lower Donchian with volume and downtrend
        if close_price < lower and vol_confirm and close_price < ema50:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite Donchian level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to lower Donchian
            exit_long = close_price < lower
        elif position == -1:
            # Exit short if price rises back to upper Donchian
            exit_short = close_price > upper
        
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