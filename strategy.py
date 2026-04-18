# %%
#!/usr/bin/env python3
"""
12h_Bollinger_Band_Width_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Bollinger Bands when Bollinger Band Width is in low volatility regime (indicating compression) with volume confirmation and EMA trend filter. Uses Bollinger Band Width percentile to identify low volatility periods, then trades breakouts from the bands. Designed to capture volatility expansion moves in both bull and bear markets with tight entry conditions. Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + bb_std * bb_std_dev
    lower_band = sma_20 - bb_std * bb_std_dev
    
    # Bollinger Band Width
    bb_width = (upper_band - lower_band) / sma_20
    
    # Bollinger Band Width percentile (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # EMA20 trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for BB width percentile
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(volume_filter[i]) or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        bb_width_pct = bb_width_percentile[i]
        vol_ok = volume_filter[i]
        ema20 = ema_20[i]
        
        if position == 0:
            # Long: price breaks above upper band in low volatility regime (BB width < 20th percentile) with volume in uptrend
            if price > upper and bb_width_pct < 20 and vol_ok and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band in low volatility regime (BB width < 20th percentile) with volume in downtrend
            elif price < lower and bb_width_pct < 20 and vol_ok and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to middle (SMA) or trend reverses
            if price < sma_20[i] or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to middle (SMA) or trend reverses
            if price > sma_20[i] or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Bollinger_Band_Width_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
# %%