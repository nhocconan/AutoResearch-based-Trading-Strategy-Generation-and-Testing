#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal (bearish) + 1d EMA(34) trend + volume spike
# Uses bearish fractal breaks for short entries in ranging/bear markets
# 1d EMA(34) provides trend filter to avoid counter-trend trades
# Volume spike confirms breakout conviction
# Designed for 12h timeframe to capture medium-term swings with low frequency
# Targets 50-150 total trades over 4 years (12-37/year)

name = "12h_WilliamsFractal_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Load 1w data ONCE before loop for Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    
    # Align bearish fractal to 12h timeframe with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Short: Price breaks below bearish fractal + price < 1d EMA(34) + volume spike
            if close[i] < bearish_fractal_aligned[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == -1:  # Short position
            # Exit: Price breaks above bearish fractal (stop and reverse conditions)
            if close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def compute_williams_fractals(high, low):
    """Compute Williams Fractals - returns (bearish, bullish) arrays"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    # Need at least 5 points: 2 left, center, 2 right
    for i in range(2, n - 2):
        # Bearish fractal: highest high with 2 lower highs on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: lowest low with 2 higher lows on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def calculate_hma(close, period):
    """Calculate Hull Moving Average - kept for potential future use"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    # WMA of full period
    wma_full = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal (bearish) + 1d EMA(34) trend + volume spike
# Uses bearish fractal breaks for short entries in ranging/bear markets
# 1d EMA(34) provides trend filter to avoid counter-trend trades
# Volume spike confirms breakout conviction
# Designed for 12h timeframe to capture medium-term swings with low frequency
# Targets 50-150 total trades over 4 years (12-37/year)

name = "12h_WilliamsFractal_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Load 1w data ONCE before loop for Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    
    # Align bearish fractal to 12h timeframe with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Short: Price breaks below bearish fractal + price < 1d EMA(34) + volume spike
            if close[i] < bearish_fractal_aligned[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == -1:  # Short position
            # Exit: Price breaks above bearish fractal (stop and reverse conditions)
            if close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def compute_williams_fractals(high, low):
    """Compute Williams Fractals - returns (bearish, bullish) arrays"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    # Need at least 5 points: 2 left, center, 2 right
    for i in range(2, n - 2):
        # Bearish fractal: highest high with 2 lower highs on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: lowest low with 2 higher lows on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def calculate_hma(close, period):
    """Calculate Hull Moving Average - kept for potential future use"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    # WMA of full period
    wma_full = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma