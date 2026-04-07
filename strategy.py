#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Bollinger Band squeeze breakout with 1-day volume confirmation and 1-day trend filter
# Long when price breaks above upper Bollinger Band (20,2) + Bollinger Width < 50th percentile (squeeze) + volume > 1.5x 20-period average + price > 50-period SMA
# Short when price breaks below lower Bollinger Band (20,2) + Bollinger Width < 50th percentile + volume > 1.5x 20-period average + price < 50-period SMA
# Exit when price crosses 20-period EMA in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day Bollinger Width percentile for squeeze detection and 1-day SMA for trend filter
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_bb_squeeze_breakout_1d_vol_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Bollinger Bands and SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20,2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # normalized width
    
    # Calculate 1-day Bollinger Width percentile (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate 1-day 50-period SMA for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # 6-period Bollinger Bands (20,2) for entry signals
    sma_20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20_6h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb_6h = sma_20_6h + 2 * std_20_6h
    lower_bb_6h = sma_20_6h - 2 * std_20_6h
    
    # 20-period EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6-hour ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_bb_6h[i]) or np.isnan(lower_bb_6h[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(sma_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 20-period EMA
            elif close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 20-period EMA
            elif close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout with squeeze, volume confirmation and trend filter
            # Squeeze filter: Bollinger Width < 50th percentile (low volatility)
            squeeze_filter = bb_width_percentile_aligned[i] < 50
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: price > 50-period SMA for long, price < 50-period SMA for short
            trend_filter_long = close[i] > sma_50_aligned[i]
            trend_filter_short = close[i] < sma_50_aligned[i]
            
            # Long: price breaks above upper Bollinger Band + squeeze + volume filter + trend filter
            if close[i] > upper_bb_6h[i] and squeeze_filter and volume_filter and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower Bollinger Band + squeeze + volume filter + trend filter
            elif close[i] < lower_bb_6h[i] and squeeze_filter and volume_filter and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals