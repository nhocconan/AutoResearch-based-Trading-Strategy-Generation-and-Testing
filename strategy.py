#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 12h EMA trend filter + ATR stoploss
# - Long: Price breaks above 20-period Donchian high + 1d volume > 1.5x 20-period average + 12h EMA(21) > EMA(50)
# - Short: Price breaks below 20-period Donchian low + same volume and trend filters
# - Exit: Close-based reversal - exit long when price < 20-period Donchian low, exit short when price > Donchian high
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Target: 75-150 total trades over 4 years (19-38/year) to stay well below HARD MAX: 400 total

name = "4h_1d_12h_donchian_breakout_volume_ema_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for EMA50 and Donchian
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 60:  # Need enough for EMA50
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for volume
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 12h data for EMA
    close_12h = df_12h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (already aligned)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 12h EMA(21) and EMA(50) for trend filter
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = 0
    
    # Wilder's ATR smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_4h = wilders_smoothing(tr_4h, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 1.5 * volume_ma_aligned[i]
        
        # Trend filter: EMA21 > EMA50 indicates uptrend, EMA21 < EMA50 indicates downtrend
        ema_trend_up = ema_21_aligned[i] > ema_50_aligned[i]
        ema_trend_down = ema_21_aligned[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + volume confirmation + uptrend
            if (close_price > donchian_high_aligned[i] and 
                volume_confirmation and 
                ema_trend_up):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume confirmation + downtrend
            elif (close_price < donchian_low_aligned[i] and 
                  volume_confirmation and 
                  ema_trend_down):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_4h[i]
                # Exit conditions: price < Donchian low OR stoploss hit
                if close_price < donchian_low_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_4h[i]
                # Exit conditions: price > Donchian high OR stoploss hit
                if close_price > donchian_high_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals