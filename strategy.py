#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Bands squeeze breakout with volume and trend confirmation
# - Uses 20-period BB width percentile to detect low volatility squeeze
# - Breakout occurs when price closes outside BB bands after squeeze
# - Trend filter: 50-period EMA on 1d timeframe (price above EMA for long, below for short)
# - Volume confirmation: volume > 1.5x 20-period average
# - Exit: price re-enters Bollinger Bands or ATR stop hit (2x ATR)
# - Designed to work in both bull and bear markets by using volatility contraction/expansion
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # 4h Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + bb_std * bb_std_dev
    bb_lower = bb_ma - bb_std * bb_std_dev
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume confirmation: 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(ema_50_4h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: BB squeeze (width < 20th percentile) + price closes above upper BB + above 1d EMA50 + volume surge
            if (bb_width_percentile[i] < 0.2 and 
                price > bb_upper[i] and 
                price > ema_50_4h[i] and 
                vol > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: BB squeeze + price closes below lower BB + below 1d EMA50 + volume surge
            elif (bb_width_percentile[i] < 0.2 and 
                  price < bb_lower[i] and 
                  price < ema_50_4h[i] and 
                  vol > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price re-enters Bollinger Bands OR ATR stop hit (2*ATR)
            if price < bb_upper[i] or price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters Bollinger Bands OR ATR stop hit (2*ATR)
            if price > bb_lower[i] or price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BB_Squeeze_Breakout_Volume_EMA50_ATRStop"
timeframe = "4h"
leverage = 1.0