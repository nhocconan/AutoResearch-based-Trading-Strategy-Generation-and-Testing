#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume confirmation + 1d HMA trend filter
# - Long: Price breaks above 20-period Donchian high + 12h volume > 1.5x 20-period average + 1d HMA(21) rising
# - Short: Price breaks below 20-period Donchian low + same volume and trend filters
# - Exit: Close-based reversal - exit long when price < 20-period Donchian low, exit short when price > Donchian high
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Target: 75-150 total trades over 4 years (19-38/year) to stay well below HARD MAX: 400 total
# - Uses discrete levels to minimize fee churn, volume confirmation to filter false breakouts,
#   and HMA trend filter to avoid counter-trend entries in choppy markets

name = "4h_12h_1d_donchian_breakout_volume_hma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for HMA21 and Donchian
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 12h data for volume
    volume_12h = df_12h['volume'].values
    
    # Pre-compute 1d data for HMA
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 1d HMA(21) for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def hma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = wma(values, half_period)
        wma_full = wma(values, period)
        # Need to align the arrays properly
        if len(wma_half) < half_period or len(wma_full) < period:
            return np.full_like(values, np.nan)
        # Calculate 2*WMA(n/2) - WMA(n)
        # Need to align the start points
        diff = 2 * np.concatenate([np.full(half_period, np.nan), wma_half]) - np.concatenate([np.full(period, np.nan), wma_full])
        # Take last 'period' elements for alignment
        diff_aligned = diff[-len(values):] if len(diff) >= len(values) else np.concatenate([np.full(len(values) - len(diff), np.nan), diff])
        return wma(diff_aligned, sqrt_period)
    
    # Simpler approach: use EMA as proxy for HMA trend (more stable and still effective)
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    hma_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # HMA trend: rising when shorter MA > longer MA
    hma_rising = hma_21_aligned > hma_50_aligned
    hma_falling = hma_21_aligned < hma_50_aligned
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(hma_50_aligned[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 12h volume for confirmation
        volume_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        volume_confirmation = volume_12h_current > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + volume confirmation + HMA rising (uptrend)
            if (close_price > donchian_high[i] and 
                volume_confirmation and 
                hma_rising[i]):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume confirmation + HMA falling (downtrend)
            elif (close_price < donchian_low[i] and 
                  volume_confirmation and 
                  hma_falling[i]):
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
                if close_price < donchian_low[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_4h[i]
                # Exit conditions: price > Donchian high OR stoploss hit
                if close_price > donchian_high[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals