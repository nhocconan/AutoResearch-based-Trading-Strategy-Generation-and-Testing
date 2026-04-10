#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout + 12h trend filter + volume confirmation
# - Primary signal: Bollinger Band width (20,2) at 6h low triggers mean reversion setup
# - Entry: Price breaks above/below Bollinger Bands with volume spike in direction of 12h trend
# - Trend filter: 12h close > EMA(50) for longs, < EMA(50) for shorts (institutional bias)
# - Volume filter: 6h volume > 1.5x 30-period average (avoid false breakouts)
# - Position size: 0.25 discrete level
# - Stoploss: 2.0x ATR(14) on 6h
# - Works in bull/bear: Bollinger squeezes precede volatility expansion in all regimes;
#   12h EMA filter ensures we trade with higher timeframe momentum

name = "6h_12h_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_trend_long = close_12h > ema_50   # Uptrend bias
    ema_trend_short = close_12h < ema_50  # Downtrend bias
    ema_trend_long_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_long)
    ema_trend_short_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_short)
    
    # Pre-compute 6h Bollinger Bands (20,2)
    close_6h = prices['close'].values
    bb_middle = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Bollinger Band squeeze: width at 20-period low (volatility contraction)
    bb_width_low = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    bb_squeeze = bb_width <= bb_width_low  # Squeeze condition
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_30 = pd.Series(volume_6h).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_30)
    
    # Pre-compute 6h ATR(14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_squeeze[i]) or np.isnan(ema_trend_long_aligned[i]) or
            np.isnan(ema_trend_short_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price closes below BB middle OR stoploss hit
            if close_6h[i] < bb_middle[i] or close_6h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above BB middle OR stoploss hit
            if close_6h[i] > bb_middle[i] or close_6h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Bollinger Band breakout with squeeze + trend + volume
            if bb_squeeze[i] and vol_spike[i]:
                # Long: Break above upper BB in uptrend
                if close_6h[i] > bb_upper[i] and ema_trend_long_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Break below lower BB in downtrend
                elif close_6h[i] < bb_lower[i] and ema_trend_short_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals