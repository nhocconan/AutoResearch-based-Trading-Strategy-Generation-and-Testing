#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d ADX trend filter and 1w volume confirmation
# - Long when price breaks above latest bearish fractal (swing high) + 1d ADX > 25 (trending) + 1w volume > 1.5x 20-period volume SMA
# - Short when price breaks below latest bullish fractal (swing low) + 1d ADX > 25 + 1w volume > 1.5x 20-period volume SMA
# - Exit: price returns to midpoint of last fractal pair (mean reversion within swing)
# - Position sizing: 0.25 discrete level
# - Williams Fractals identify significant swing points, ADX filters for trending markets, volume confirms institutional participation
# - Works in bull/bear: breakouts work in strong trends, mean reversion exit works in ranging periods
# - 6h timeframe targets 12-37 trades/year with strict entry conditions to minimize fee drag

name = "6h_1w_1d_fractal_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d ADX(14) for trend filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed TR, +DM, -DM
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr != 0, atr, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.where(atr != 0, atr, 1e-10)
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (using completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1w volume SMA(20) for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Calculate Williams Fractals on 1d data (requires 5-bar window: n-2, n-1, n, n+1, n+2)
    # Bearish fractal (swing high): high[n] > high[n-1] and high[n] > high[n-2] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal (swing low): low[n] < low[n-1] and low[n] < low[n-2] and low[n] < low[n+1] and low[n] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals to 6h timeframe with extra delay (fractals need 2 extra 1d bars for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Track latest confirmed fractal levels
    latest_bearish = np.full(n, np.nan)
    latest_bullish = np.full(n, np.nan)
    last_bearish = np.nan
    last_bullish = np.nan
    
    for i in range(n):
        if not np.isnan(bearish_fractal_aligned[i]):
            last_bearish = bearish_fractal_aligned[i]
        if not np.isnan(bullish_fractal_aligned[i]):
            last_bullish = bullish_fractal_aligned[i]
        latest_bearish[i] = last_bearish
        latest_bullish[i] = last_bullish
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20_1w_aligned[i]) or 
            np.isnan(latest_bearish[i]) or np.isnan(latest_bullish[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1w volume for volume spike confirmation
        vol_1w_current = align_htf_to_ltf(prices, df_1w, df_1w['volume'].values)
        
        # Volume confirmation: current 1w volume > 1.5x 20-period SMA (volume spike)
        vol_confirm = vol_1w_current[i] > 1.5 * volume_sma_20_1w_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market (favorable for breakouts)
        trending_market = adx_aligned[i] > 25
        
        # Fractal breakout signals
        breakout_up = close[i] > latest_bearish[i]   # Price breaks above latest bearish fractal (swing high)
        breakout_down = close[i] < latest_bullish[i] # Price breaks below latest bullish fractal (swing low)
        
        # Exit: price returns to midpoint of last fractal pair (mean reversion within swing)
        # Only calculate midpoint if both fractals are available
        if not np.isnan(latest_bearish[i]) and not np.isnan(latest_bullish[i]):
            fractal_midpoint = (latest_bearish[i] + latest_bullish[i]) / 2
            return_to_midpoint = abs(close[i] - fractal_midpoint) < (latest_bearish[i] - latest_bullish[i]) * 0.2  # Within 20% of swing range
        else:
            return_to_midpoint = False
        
        # Entry conditions: Fractal breakout with volume and trend confirmation
        long_entry = breakout_up and vol_confirm and trending_market
        short_entry = breakout_down and vol_confirm and trending_market
        
        # Exit conditions: price returns to midpoint of fractal pair
        long_exit = return_to_midpoint  # Exit long when price returns to midpoint
        short_exit = return_to_midpoint  # Exit short when price returns to midpoint
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals