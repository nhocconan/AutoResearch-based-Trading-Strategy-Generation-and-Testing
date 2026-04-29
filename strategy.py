#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and ATR-based position sizing
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# Weekly EMA50 trend filter ensures we only trade in direction of higher timeframe trend
# ATR-based position sizing adjusts for volatility - smaller size in high vol, larger in low vol
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# Target: 15-25 trades/year (60-100 total over 4 years)

name = "1d_Donchian20_WeeklyTrend_ATRSize_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on daily timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility-based position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First period TR
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Calculate dynamic position size based on ATR (inverse volatility)
        # Base size 0.30 scaled by ATR relative to its 50-period median
        if i >= 50:
            atr_median = np.nanmedian(atr[max(0, i-49):i+1])
            if atr_median > 0:
                vol_scaling = min(1.5, max(0.5, atr_median / curr_atr))  # Inverse vol scaling
                position_size = 0.30 * vol_scaling
                position_size = min(0.40, max(0.10, position_size))  # Clamp to reasonable range
            else:
                position_size = 0.30
        else:
            position_size = 0.30
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above Donchian high AND price above weekly EMA50 (uptrend)
            if curr_close > curr_donchian_high and curr_close > curr_ema_50_1w:
                signals[i] = position_size
                position = 1
            # Short breakdown: price breaks below Donchian low AND price below weekly EMA50 (downtrend)
            elif curr_close < curr_donchian_low and curr_close < curr_ema_50_1w:
                signals[i] = -position_size
                position = -1
        
        elif position == 1:  # Long position - exit when price breaks below Donchian low
            if curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        
        elif position == -1:  # Short position - exit when price breaks above Donchian high
            if curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals