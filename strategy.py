# Hypothetical solution: 6h timeframe, novel concept – adaptive volatility breakout with regime filter (volatility regime + ATR breakout + 12h trend filter)
# The idea: In low volatility regimes, a breakout of a volatility-adjusted Donchian channel (using ATR-scaled bands) combined with 12h trend alignment has edge.
# This avoids the saturated pure price-channel breakouts by incorporating volatility regime detection.
# We target 50-150 total trades over 4 years by using a volatility filter that reduces trades in chop.
# The strategy should work in both bull and bear because it follows the 12h trend and only takes breakouts in the trend direction.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and volatility regime
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h ATR(20) for volatility regime (using 12h high/low/close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr20_12h = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime: current 12h ATR relative to its 50-period average (expanding/contracting vol)
    atr50_avg_12h = pd.Series(atr20_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_ratio = atr20_12h / atr50_avg_12h  # >1 = expanding vol, <1 = contracting vol
    # We want to trade breakouts when volatility is expanding (vol_ratio > 1.0) to catch real moves
    vol_expanding = vol_ratio > 1.0
    
    # Volatility-adjusted Donchian channel on 6h (using ATR to scale bands)
    # We'll use a 20-period lookback for the channel, but scaled by ATR to normalize for volatility
    # Instead, we compute a normalized channel: (price - rolling mean) / ATR, then breakout when normalized > threshold
    # Simpler: compute ATR-scaled breakout bands: upper = rolling max(high) + k*ATR, lower = rolling min(low) - k*ATR
    # Use k=0.5 to avoid too many breakouts.
    lookback = 20
    k = 0.5
    # Rolling max/min of high/low
    roll_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Current 6h ATR(20) for scaling
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    upper_band = roll_max + k * atr
    lower_band = roll_min - k * atr
    
    # Align all indicators to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_expanding_aligned = align_htf_to_ltf(prices, df_12h, vol_expanding)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)  # Note: upper_band is 6s? Actually computed on 6h, but we align anyway for consistency (though not needed)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # However, upper_band and lower_band are computed on 6h data, so they are already LTF. 
    # To follow the rule of using HTF data for filters, we instead compute the breakout bands on 12h and align to 6h.
    # Let's recompute: use 12h data to build volatility-adjusted Donchian, then align to 6h.
    # This ensures we are using HTF for the breakout levels.
    roll_max_12h = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    roll_min_12h = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    upper_band_12h = roll_max_12h + k * atr20_12h
    lower_band_12h = roll_min_12h - k * atr20_12h
    upper_band_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_band_12h)
    lower_band_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_band_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level to reduce turnover)
    
    # Warmup: need EMA50 (50), ATR20 (20), vol_ratio (50 for EMA of ATR), Donchian (20)
    start_idx = max(50, 50, 20, 20)  # at least 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_expanding_aligned[i]) or 
            np.isnan(upper_band_12h_aligned[i]) or np.isnan(lower_band_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_12h_aligned[i]
        vol_exp = vol_expanding_aligned[i]
        upper = upper_band_12h_aligned[i]
        lower = lower_band_12h_aligned[i]
        
        if position == 0:
            # Determine trend: price vs 12h EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_exp:
                # Long breakout: price above upper band
                if close_val > upper:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_exp:
                # Short breakdown: price below lower band
                if close_val < lower:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: reverse signal or volatility contraction (vol_expanding false) as risk management
            if not vol_exp or close_val < ema50:  # trend change or vol contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            if not vol_exp or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_VolRegime_Breakout_12hTrend"
timeframe = "6h"
leverage = 1.0