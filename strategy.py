#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter + volume confirmation
# - Donchian(20) breakout provides clean entry/exit signals with low trade frequency
# - 1-week EMA50 trend filter ensures we only trade in direction of higher timeframe trend
# - Volume confirmation (1.5x 20-period average) filters out weak breakouts
# - Discrete position sizing ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1w['volume'].values if 'volume' in df_1w.columns else np.ones(len(df_1w))  # fallback if no volume
    # Actually get 1d volume data for proper confirmation
    df_1d_for_vol = get_htf_data(prices, '1d')
    if len(df_1d_for_vol) >= 20:
        volume_1d_actual = df_1d_for_vol['volume'].values
        volume_sma_20_1d = pd.Series(volume_1d_actual).rolling(window=20, min_periods=20).mean().values
        volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d_for_vol, volume_sma_20_1d)
    else:
        volume_sma_20_aligned = np.ones(n) * 1e-9  # avoid division by zero
    
    # Pre-compute 1d Donchian channels (20-period)
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = close_current > donchian_high[i-1]  # break above previous period's high
        breakout_short = close_current < donchian_low[i-1]  # break below previous period's low
        
        # 1-week trend filter: price above/below EMA50
        uptrend = close_current > ema50_1w_aligned[i]
        downtrend = close_current < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish breakout + uptrend + volume confirmation
        if breakout_long and uptrend and vol_confirm:
            enter_long = True
        
        # Short: bearish breakout + downtrend + volume confirmation
        if breakout_short and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse Donchian breakout or trend change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below Donchian low OR trend turns down
            exit_long = (close_current < donchian_low[i]) or (not uptrend)
        elif position == -1:
            # Exit short if price breaks above Donchian high OR trend turns up
            exit_short = (close_current > donchian_high[i]) or (not downtrend)
        
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