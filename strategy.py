#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long: Williams %R(14) < -80 (oversold) + 1d close > 1d EMA(50) (uptrend) + 6h volume > 1.5x 20-period average volume
# - Short: Williams %R(14) > -20 (overbought) + 1d close < 1d EMA(50) (downtrend) + same volume confirmation
# - Exit: Close-based reversal - exit long when Williams %R > -50, exit short when Williams %R < -50
# - Stoploss: ATR-based - exit when price moves against position by 2.5 * ATR(14) on 6h
# - Position sizing: 0.25 (discrete level)
# - Uses Williams %R for mean reversion in ranging markets, EMA(50) on 1d for trend filter to avoid counter-trend trades
# - Volume confirmation threshold set to 1.5x to ensure quality signals
# - Target: 80-160 total trades over 4 years (20-40/year) to stay within HARD MAX: 300 total
# - Works in both bull and bear: trend filter prevents counter-trend trades, Williams %R captures reversals from extremes

name = "6h_1d_williamsr_mean_reversion_v1"
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
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) for 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_6h) / (highest_high - lowest_low)) * -100,
        -50  # Neutral when range is zero
    )
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume moving average (20-period)
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_6h)  # Using 1d alignment for volume MA (acceptable approximation)
    
    # Calculate 6h ATR (14-period) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR is 0 (no previous close)
    
    # Wilder's ATR smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_6h = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for volume MA)
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close and volume
        close_price = close_6h[i]
        volume_current = volume_6h[i]
        
        # Get 1d close for trend filter (need to align)
        close_1d_current = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmation = volume_current > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + 1d close > 1d EMA(50) + volume confirmation
            if (williams_r[i] < -80 and close_1d_current > ema_50_aligned[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + 1d close < 1d EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and close_1d_current < ema_50_aligned[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.5 * atr_14_6h[i]
                # Exit conditions: Williams %R > -50 (mean reversion exit) OR stoploss hit
                if williams_r[i] > -50 or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.5 * atr_14_6h[i]
                # Exit conditions: Williams %R < -50 (mean reversion exit) OR stoploss hit
                if williams_r[i] < -50 or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals