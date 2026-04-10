#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR expansion + 1w EMA trend filter
# - Primary signal: Price breaks above/below 20-period Donchian channel on 4h
# - Volatility filter: 1d ATR(14) > 1.2x its 50-period EMA (ensures breakouts occur during expanding volatility)
# - Trend filter: 1w EMA(50) slope positive for longs, negative for shorts (avoids counter-trend trades)
# - Works in bull/bear: In bull markets, longs taken when 1w EMA up; in bear markets, shorts taken when 1w EMA down
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "4h_1d_1w_donchian_atr_ema_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d ATR expansion filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d1[0] if 'tr_1d1' in locals() else tr_1d[0]  # Will be fixed below
    
    # Fix: compute tr components properly
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first period TR
    
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_ema = pd.Series(atr_14).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_expansion = atr_14 > (1.2 * atr_50_ema)
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    
    # Pre-compute 1w EMA(50) slope for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA slope: positive if current EMA > previous EMA
    ema_slope = np.zeros_like(ema_50)
    ema_slope[1:] = ema_50[1:] > ema_50[:-1]
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    # Pre-compute 4h Donchian Channel (20)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_expansion_aligned[i]) or np.isnan(ema_slope_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] < (donchian_high[i] + donchian_low[i]) / 2 or close_4h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] > (donchian_high[i] + donchian_low[i]) / 2 or close_4h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with ATR expansion and EMA trend filter
            # Long: break above upper band with ATR expansion and rising 1w EMA
            if close_4h[i] > donchian_high[i] and atr_expansion_aligned[i] and ema_slope_aligned[i]:
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short: break below lower band with ATR expansion and falling 1w EMA
            elif close_4h[i] < donchian_low[i] and atr_expansion_aligned[i] and not ema_slope_aligned[i]:
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
    
    return signals