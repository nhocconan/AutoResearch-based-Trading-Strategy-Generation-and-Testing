#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter, volume confirmation, and ATR stop
# Works in bull markets via breakout momentum; works in bear markets via trend filter preventing false breakouts
# Target: 20-40 trades/year (80-160 total over 4 years)
name = "4h_Donchian20_1dTrend_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h: Price and volume data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h: Donchian channel (20-period) ===
    # Use rolling window with min_periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 4h: ATR for stop calculation ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        ema_trend = ema_50_aligned[i]
        
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_val) or np.isnan(ema_trend)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, uptrend (price > EMA50), volume confirmation
            if (close_val > donchian_high_val and  # Breakout above resistance
                close_val > ema_trend and          # Uptrend filter
                vol_ratio_val > 1.8):              # Volume confirmation
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Price breaks below Donchian low, downtrend (price < EMA50), volume confirmation
            elif (close_val < donchian_low_val and   # Breakdown below support
                  close_val < ema_trend and          # Downtrend filter
                  vol_ratio_val > 1.8):              # Volume confirmation
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long position management
            # Exit conditions: price drops below Donchian low OR hits ATR-based stop
            if (close_val < donchian_low_val or 
                close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position management
            # Exit conditions: price rises above Donchian high OR hits ATR-based stop
            if (close_val > donchian_high_val or 
                close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals