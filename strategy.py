# Adaptive Trend & Mean Reversion with 1D ATR Filter for 6h Timeframe
# Hypothesis: In trending markets (bull/bear), use adaptive EMA crossover for momentum capture.
# In ranging markets, use ATR-based mean reversion at Bollinger Band extremes.
# Uses 1D ATR to dynamically adjust sensitivity and filter trades across regimes.
# Designed to work in both bull (trend-following) and bear (mean-reverting) markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_AdaptiveTrendMeanReversion_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Get 1D data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1D Indicators: ATR(14) for volatility regime filtering ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # === Align 1D ATR to 6h timeframe ===
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 6H Indicators: Price and EMAs for adaptive strategy ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Adaptive EMAs: faster in low vol, slower in high vol
    # Base periods adjusted by volatility regime
    close_series = pd.Series(close)
    
    # Calculate volatility regime using 1D ATR ratio (current vs 50-period average)
    atr_ma50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
    vol_ratio = atr_1d_aligned / np.where(atr_ma50 > 0, atr_ma50, np.nan)
    
    # Adaptive EMA periods: 10-30 range based on volatility
    # Low vol (ratio < 0.8): faster EMAs (12, 26)
    # High vol (ratio > 1.2): slower EMAs (18, 39)
    # Medium vol: linear interpolation
    ema_fast_base = 12
    ema_slow_base = 26
    
    # Clamp vol ratio for stability
    vol_ratio_clamped = np.clip(vol_ratio, 0.5, 2.0)
    
    # Calculate adaptive periods
    ema_fast_period = ema_fast_base + (ema_fast_base * 0.5) * (vol_ratio_clamped - 1.0)
    ema_slow_period = ema_slow_base + (ema_slow_base * 0.5) * (vol_ratio_clamped - 1.0)
    
    # Ensure minimum periods
    ema_fast_period = np.maximum(ema_fast_period, 8)
    ema_slow_period = np.maximum(ema_slow_period, 16)
    
    # Calculate EMAs with adaptive periods (using EMA formula with variable alpha)
    ema_fast = np.zeros_like(close)
    ema_slow = np.zeros_like(close)
    
    # Seed EMAs
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    for i in range(1, len(close)):
        if not np.isnan(ema_fast_period[i]) and ema_fast_period[i] > 0:
            alpha_fast = 2.0 / (ema_fast_period[i] + 1)
        else:
            alpha_fast = 2.0 / (ema_fast_base + 1)
            
        if not np.isnan(ema_slow_period[i]) and ema_slow_period[i] > 0:
            alpha_slow = 2.0 / (ema_slow_period[i] + 1)
        else:
            alpha_slow = 2.0 / (ema_slow_base + 1)
            
        ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i-1]
        ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i-1]
    
    # === Bollinger Bands (20, 2) for mean reversion signals ===
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any critical data is NaN
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        ema_fast_val = ema_fast[i]
        ema_slow_val = ema_slow[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Determine regime: low volatility favors mean reversion, high volatility favors trend following
            # Vol ratio < 0.9: low vol (range) -> mean reversion
            # Vol ratio > 1.1: high vol (trend) -> trend following
            # Middle: blended approach
            
            if vol_ratio_val < 0.9:
                # Low volatility regime: mean reversion at Bollinger extremes
                if close_val <= bb_lower_val:
                    # Oversold - go long
                    signals[i] = 0.25
                    position = 1
                elif close_val >= bb_upper_val:
                    # Overbought - go short
                    signals[i] = -0.25
                    position = -1
            elif vol_ratio_val > 1.1:
                # High volatility regime: trend following with EMA crossover
                if ema_fast_val > ema_slow_val:
                    # Uptrend - go long
                    signals[i] = 0.25
                    position = 1
                elif ema_fast_val < ema_slow_val:
                    # Downtrend - go short
                    signals[i] = -0.25
                    position = -1
            else:
                # Medium volatility: require both conditions for entry
                # Trend alignment + volatility confirmation
                if ema_fast_val > ema_slow_val and close_val > bb_middle[i]:
                    # Uptrend confirmation
                    signals[i] = 0.20
                    position = 1
                elif ema_fast_val < ema_slow_val and close_val < bb_middle[i]:
                    # Downtrend confirmation
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if vol_ratio_val < 0.9:
                # Mean reversion exit: return to middle
                if close_val >= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif vol_ratio_val > 1.1:
                # Trend following exit: EMA cross down
                if ema_fast_val < ema_slow_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Medium volatility: exit on mean reversion or trend break
                if close_val >= bb_middle[i] or ema_fast_val < ema_slow_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if vol_ratio_val < 0.9:
                # Mean reversion exit: return to middle
                if close_val <= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif vol_ratio_val > 1.1:
                # Trend following exit: EMA cross up
                if ema_fast_val > ema_slow_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Medium volatility: exit on mean reversion or trend break
                if close_val <= bb_middle[i] or ema_fast_val > ema_slow_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals