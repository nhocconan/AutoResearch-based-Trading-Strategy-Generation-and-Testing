# Hypothesis: 4h Williams Fractal breakout with volume confirmation and 1d EMA trend filter
# Williams Fractal identifies pivot points. Breakout above/below recent fractal with volume
# and trend filter should capture momentum moves. Designed for fewer trades (target 20-50/year)
# to avoid fee drag. Works in bull/bear via trend filter and volume confirmation.
# Target: 25-40 trades/year per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest of 5 bars
        if high[i] >= high[i-2] and high[i] >= high[i-1] and high[i] >= high[i+1] and high[i] >= high[i+2]:
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if low[i] <= low[i-2] and low[i] <= low[i-1] and low[i] <= low[i+1] and low[i] <= low[i+2]:
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals on 1d
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    # Williams fractal needs 2 extra bars for confirmation (center + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d average volume (20-period) for volume confirmation
    vol_ma_1d = np.zeros_like(volume_1d)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d[:20] = np.nan
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if NaN in critical values
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current 4h values
        price = close[i]
        vol = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Get recent fractal values (lookback up to 20 days for recent fractal)
        lookback_start = max(0, i - 20 * 24)  # ~20 days of 4h bars
        recent_bearish = bearish_fractal_aligned[lookback_start:i+1]
        recent_bullish = bullish_fractal_aligned[lookback_start:i+1]
        
        # Find most recent valid fractal
        last_bearish = np.nanmax(recent_bearish) if not np.all(np.isnan(recent_bearish)) else np.nan
        last_bullish = np.nanmin(recent_bullish) if not np.all(np.isnan(recent_bullish)) else np.nan
        
        if position == 0:
            # Long: price breaks above recent bearish fractal (resistance) with volume + uptrend
            if (not np.isnan(last_bearish) and price > last_bearish and 
                ema_trend > 0 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent bullish fractal (support) with volume + downtrend
            elif (not np.isnan(last_bullish) and price < last_bullish and 
                  ema_trend < 0 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below recent bullish fractal (support) or trend change
            if (not np.isnan(last_bullish) and price < last_bullish) or ema_trend < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above recent bearish fractal (resistance) or trend change
            if (not np.isnan(last_bearish) and price > last_bearish) or ema_trend > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0