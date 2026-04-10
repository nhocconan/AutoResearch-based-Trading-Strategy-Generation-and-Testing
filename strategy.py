#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
# - Long: price breaks above Donchian upper(20) + price > EMA(34) on 1w + 1d volume > 1.3x 20-period average volume
# - Short: price breaks below Donchian lower(20) + price < EMA(34) on 1w + same volume confirmation
# - Exit: Close-based reversal - exit long when price < Donchian middle(20), exit short when price > Donchian middle(20)
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 1d
# - Position sizing: 0.25 (discrete level)
# - Uses Donchian channels for breakout structure, 1w EMA for higher-timeframe trend filter to avoid counter-trend trades
# - Volume confirmation ensures participation
# - Target: 40-80 total trades over 4 years (10-20/year) to stay within HARD MAX: 150 total
# - Works in both bull and bear: breakouts capture strong moves, trend filter avoids false breakouts in chop

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian Channel (20-period) for 1d
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2.0
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1d)
    
    # Calculate 1d ATR (14-period) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
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
    
    atr_14_1d = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for Donchian)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_14_1d[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close and volume
        close_price = close_1d[i]
        volume_price = volume_1d[i]
        volume_confirmation = volume_price > 1.3 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + price > EMA(34) 1w + volume confirmation
            if (close_price > highest_high[i] and close_price > ema_34_aligned[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower + price < EMA(34) 1w + volume confirmation
            elif (close_price < lowest_low[i] and close_price < ema_34_aligned[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_1d[i]
                # Exit conditions: price < Donchian middle OR stoploss hit
                if close_price < donchian_middle[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_1d[i]
                # Exit conditions: price > Donchian middle OR stoploss hit
                if close_price > donchian_middle[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals