#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and 1d volume spike filter
# - Entry: Long when price breaks above 4h Donchian upper channel (20) + 1d volume > 2.0x 20-period average + 12h EMA(50) > EMA(200)
#          Short when price breaks below 4h Donchian lower channel (20) + same volume and trend filters
# - Exit: Close-based reversal - exit long when price < 4h Donchian lower channel, exit short when price > 4h Donchian upper channel
# - Stoploss: ATR-based - exit when price moves against position by 2.5 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Volume spike filter (2.0x average) is stricter than typical 1.5x to reduce false breakouts and trade frequency
# - EMA crossover (50 > 200) provides strong trend filter that works in both bull and bear markets
# - Target: 75-150 total trades over 4 years (19-38/year) to stay well within HARD MAX: 400 total

name = "4h_1d_12h_donchian_breakout_volume_ema_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for EMA200
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 200:  # Need enough for EMA200
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
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 12h EMA(50) and EMA(200) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_12h = close_12h_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
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
    
    for i in range(200, n):  # Start after warmup period for EMA200
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 2.0 * volume_ma_aligned[i]
        
        # Trend filter: EMA50 > EMA200 indicates uptrend, EMA50 < EMA200 indicates downtrend
        ema_trend_up = ema_50_aligned[i] > ema_200_aligned[i]
        ema_trend_down = ema_50_aligned[i] < ema_200_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper channel + volume confirmation + uptrend
            if (close_price > donchian_high_4h[i] and 
                volume_confirmation and 
                ema_trend_up):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower channel + volume confirmation + downtrend
            elif (close_price < donchian_low_4h[i] and 
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
                stop_loss = entry_price - 2.5 * atr_14_4h[i]
                # Exit conditions: price < Donchian lower channel OR stoploss hit
                if close_price < donchian_low_4h[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.5 * atr_14_4h[i]
                # Exit conditions: price > Donchian upper channel OR stoploss hit
                if close_price > donchian_high_4h[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals