#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h ADX trend filter and volume confirmation
# - Long: BB Width(20) < 20th percentile (squeeze) + price breaks above upper BB + 12h ADX > 25 + 12h volume > 1.5x 20-period MA
# - Short: BB Width(20) < 20th percentile (squeeze) + price breaks below lower BB + 12h ADX > 25 + 12h volume > 1.5x 20-period MA
# - Exit: Price returns to middle BB (20-period SMA) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# - Bollinger Band squeeze identifies low volatility primed for breakout; ADX ensures breakout has trend strength
# - Volume confirmation filters weak breakouts; works in both bull (continuation) and bear (mean reversion traps) markets

name = "6h_12h_bb_squeeze_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Bollinger Bands (20, 2) for 6h
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20 * 100  # Percentage width
    
    # Calculate BB Width percentile rank (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile = np.where(np.isnan(bb_width_percentile), 50, bb_width_percentile)
    
    # Squeeze condition: BB Width < 20th percentile
    squeeze = bb_width_percentile < 20
    
    # Breakout conditions
    breakout_up = close_6h > upper_bb
    breakout_down = close_6h < lower_bb
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_12h - np.roll(high_12h, 1))
    down_move = pd.Series(np.roll(low_12h, 1) - low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr_14 == 0, 1, atr_14)
    minus_di = 100 * minus_dm_smooth / np.where(atr_14 == 0, 1, atr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_12h, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_12h, minus_di)
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        squeeze_current = squeeze[i]
        breakout_up_current = breakout_up[i]
        breakout_down_current = breakout_down[i]
        adx_current = adx_aligned[i]
        plus_di_current = plus_di_aligned[i]
        minus_di_current = minus_di_aligned[i]
        volume_12h_current = volume_12h_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_12h_current > 1.5 * volume_ma_current
        
        # Trend direction from DI crossover
        uptrend = plus_di_current > minus_di_current
        downtrend = minus_di_current > plus_di_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: squeeze + breakout up + ADX > 25 + volume spike + uptrend bias
            if (squeeze_current and breakout_up_current and 
                adx_current > 25 and volume_spike and uptrend):
                position = 1
                signals[i] = 0.25
            # Short entry: squeeze + breakout down + ADX > 25 + volume spike + downtrend bias
            elif (squeeze_current and breakout_down_current and 
                  adx_current > 25 and volume_spike and downtrend):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to middle BB (20-period SMA) or opposite signal
            if position == 1:  # Long position
                if close_6h[i] <= sma_20[i]:  # Exit long when price crosses below middle BB
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (short position)
                if close_6h[i] >= sma_20[i]:  # Exit short when price crosses above middle BB
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals