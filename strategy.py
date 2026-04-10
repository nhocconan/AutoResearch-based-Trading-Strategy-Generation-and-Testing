#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss
# - Long: Price breaks above 20-period Donchian high + 1d volume > 1.8x 20-period average volume
# - Short: Price breaks below 20-period Donchian low + same volume confirmation
# - Exit: Close-based reversal - exit long when price < Donchian low, exit short when price > Donchian high
# - Stoploss: ATR-based - exit when price moves against position by 2.5 * ATR(14) on 4h
# - Position sizing: 0.30 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within HARD MAX: 400 total
# - Uses Donchian channels from 4h data, volume confirmation from daily HTF to filter false breakouts,
#   and ATR stoploss for risk management. Daily HTF volume ensures institutional participation.

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for volume
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channel (20-period) for 4h
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
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
    
    atr_14_4h = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for Donchian)
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 1.8 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + volume confirmation
            if (close_price > donchian_high[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.30
            # Short entry: price breaks below Donchian low + volume confirmation
            elif (close_price < donchian_low[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.5 * atr_14_4h[i]
                # Exit conditions: price < Donchian low OR stoploss hit
                if close_price < donchian_low[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.5 * atr_14_4h[i]
                # Exit conditions: price > Donchian high OR stoploss hit
                if close_price > donchian_high[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals