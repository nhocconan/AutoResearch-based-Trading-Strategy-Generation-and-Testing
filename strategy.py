#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout + 1d trend filter + volume confirmation
# - Long when price breaks above recent bearish fractal AND 1d close > 1d SMA50 AND volume > 1.3x 20-period average
# - Short when price breaks below recent bullish fractal AND 1d close < 1d SMA50 AND volume > 1.3x 20-period average
# - Exit when price crosses 20-period EMA OR opposite fractal breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams fractals provide high-probability reversal/breakout levels
# - 1d SMA50 filter ensures we trade with the daily trend
# - Volume confirmation reduces false breakouts
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_fractal_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Pre-compute 4h 20-period EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Pre-compute 1d trend filter: close > SMA50
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    daily_uptrend = close_1d > sma_50_1d
    daily_downtrend = close_1d < sma_50_1d
    
    # Align 1d trend to 4h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)
    
    # Compute Williams Fractals on 4h data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bearish fractal (peak)
        if (high[i-2] < high[i-1] and 
            high[i] < high[i-1] and 
            high[i-1] > high[i-3] and 
            high[i-1] > high[i+1]):
            bearish_fractal[i-1] = high[i-1]
        
        # Bullish fractal (trough)
        if (low[i-2] > low[i-1] and 
            low[i] > low[i-1] and 
            low[i-1] < low[i-3] and 
            low[i-1] < low[i+1]):
            bullish_fractal[i-1] = low[i-1]
    
    # Forward fill fractal levels to use as breakout levels
    bearish_fractal_ff = np.full(n, np.nan)
    bullish_fractal_ff = np.full(n, np.nan)
    
    last_bear = np.nan
    last_bull = np.nan
    for i in range(n):
        if not np.isnan(bearish_fractal[i]):
            last_bear = bearish_fractal[i]
        if not np.isnan(bullish_fractal[i]):
            last_bull = bullish_fractal[i]
        bearish_fractal_ff[i] = last_bear
        bullish_fractal_ff[i] = last_bull
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_ff[i]) or np.isnan(bullish_fractal_ff[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(daily_downtrend_aligned[i]) or np.isnan(ema_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above recent bearish fractal AND daily uptrend AND volume spike
            if (close[i] > bearish_fractal_ff[i] and 
                daily_uptrend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below recent bullish fractal AND daily downtrend AND volume spike
            elif (close[i] < bullish_fractal_ff[i] and 
                  daily_downtrend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses 20-period EMA OR opposite fractal breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < ema_20[i] or close[i] < bullish_fractal_ff[i]))
            exit_short = (position == -1 and 
                         (close[i] > ema_20[i] or close[i] > bearish_fractal_ff[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals