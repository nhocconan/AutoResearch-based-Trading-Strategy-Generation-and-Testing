#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 12h/1d regime filter
# - Primary: 6h timeframe for lower trade frequency and reduced fee drag
# - HTF: 12h for trend direction (EMA21), 1d for Williams Fractal confirmation
# - Long: Price breaks above recent bearish fractal high + 12h EMA21 uptrend + 1d bullish fractal intact
# - Short: Price breaks below recent bullish fractal low + 12h EMA21 downtrend + 1d bearish fractal intact
# - Exit: Price reverts to 6h EMA50 or breaks opposite fractal level
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Works in bull/bear: Fractals capture swing points; EMA filter avoids counter-trend traps

name = "6h_12h_1d_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pre-compute 1d data for Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA21 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_21_12h = close_12h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 6h EMA50 for exit
    close_6h_series = pd.Series(close_6h)
    ema_50_6h = close_6h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Fractals on 1d (requires 5-bar window: n-2, n-1, n, n+1, n+2)
    # Bearish fractal: high[n] is highest among high[n-2:n+3]
    # Bullish fractal: low[n] is lowest among low[n-2:n+3]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal: current high is highest of 5 bars
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: current low is lowest of 5 bars
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra 1d bars for confirmation (center bar + 2 future bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 6h rolling max/min for breakout levels (using recent 20 bars)
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    rolling_max_20 = high_6h_series.rolling(window=20, min_periods=20).max().values
    rolling_min_20 = low_6h_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(ema_50_6h[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(rolling_max_20[i]) or np.isnan(rolling_min_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above recent 6h high + 12h EMA21 uptrend + 1d bullish fractal intact
            # 12h EMA21 uptrend: current price > EMA21
            # 1d bullish fractal intact: bullish fractal level is valid (not NaN)
            if (close_6h[i] > rolling_max_20[i] and 
                close_6h[i] > ema_21_12h_aligned[i] and 
                not np.isnan(bullish_fractal_aligned[i])):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below recent 6h low + 12h EMA21 downtrend + 1d bearish fractal intact
            # 12h EMA21 downtrend: current price < EMA21
            # 1d bearish fractal intact: bearish fractal level is valid (not NaN)
            elif (close_6h[i] < rolling_min_20[i] and 
                  close_6h[i] < ema_21_12h_aligned[i] and 
                  not np.isnan(bearish_fractal_aligned[i])):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to 6h EMA50 (mean reversion)
            # 2. Price breaks opposite recent 6h extreme (stop loss)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_6h[i] < ema_50_6h[i] or  # Reverted to EMA50
                    close_6h[i] < rolling_min_20[i]  # Break below recent low (stop loss)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_6h[i] > ema_50_6h[i] or  # Reverted to EMA50
                    close_6h[i] > rolling_max_20[i]  # Break above recent high (stop loss)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals