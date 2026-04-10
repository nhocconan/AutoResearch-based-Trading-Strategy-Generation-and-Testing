#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based volatility breakout with 1d trend filter and volume confirmation
# - Long: 6h close > 6h open + 1.5 * ATR(14) + price > 1d EMA200 (uptrend) + 1d volume > 1.3x 20-period MA
# - Short: 6h open > 6h close + 1.5 * ATR(14) + price < 1d EMA200 (downtrend) + 1d volume > 1.3x 20-period MA
# - Exit: Close back inside the ATR band (mean reversion) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag
# - Volatility breakouts capture momentum after consolidation, EMA200 filters for higher-timeframe trend,
#   volume confirms institutional participation. Works in bull/bear: breakouts with trend in bull,
#   mean reversion exits in bear ranges.

name = "6h_1d_atr_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR (14-period) for 6h
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA200
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for ATR14)
        # Skip if any required data is invalid
        if (np.isnan(atr_14[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h OHLC
        open_price = open_6h[i]
        close_price = close_6h[i]
        
        # Get aligned 1d data for current 6h bar (completed 1d bar)
        ema_200_current = ema_200_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume spike condition: current 1d volume > 1.3x 20-period MA
        volume_spike = volume_1d_current > 1.3 * volume_ma_current
        
        # ATR breakout conditions
        body_size = np.abs(close_price - open_price)
        atr_threshold = 1.5 * atr_14[i]
        
        bullish_breakout = close_price > open_price and body_size > atr_threshold
        bearish_breakout = open_price > close_price and body_size > atr_threshold
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish ATR breakout + price > 1d EMA200 + volume spike
            if (bullish_breakout and close_price > ema_200_current and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish ATR breakout + price < 1d EMA200 + volume spike
            elif (bearish_breakout and close_price < ema_200_current and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price closes back inside the ATR band (mean reversion)
            if position == 1 and close_price < open_price + atr_threshold:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close_price > open_price - atr_threshold:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals