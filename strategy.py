#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA(50) trend filter and volume confirmation
# - Long: Price breaks above H3 level + 4h EMA(50) rising + 1h volume > 1.5x 20-period average volume
# - Short: Price breaks below L3 level + 4h EMA(50) falling + same volume confirmation
# - Exit: Close-based reversal - exit long when price < L3, exit short when price > H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 1h
# - Position sizing: 0.20 (discrete level)
# - Uses Camarilla pivots from previous 4h bar for structure, 4h EMA for trend filter to avoid counter-trend trades
# - Volume confirmation threshold set to 1.5x to reduce false breakouts
# - Session filter: 08-20 UTC to reduce noise trades
# - Target: 60-150 total trades over 4 years (15-37/year) to stay within HARD MAX: 200 total
# - Works in both bull and bear: trend filter prevents counter-trend trades, Camarilla breakouts capture momentum

name = "1h_4h_camarilla_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 1h OHLCV
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute 4h data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_series = pd.Series(close_4h)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = close_4h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h EMA slope (rising/falling)
    ema_slope = np.diff(ema_50_aligned, prepend=np.nan)
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Calculate 1h ATR (14-period) for stoploss
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
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
    
    atr_14_1h = wilders_smoothing(tr, 14)
    
    # Calculate 1h volume moving average (20-period)
    volume_1h_series = pd.Series(volume_1h)
    volume_ma_20_1h = volume_1h_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for volume MA)
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_1h[i]) or 
            np.isnan(volume_ma_20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivots from previous 4h bar
        # Need to get the previous completed 4h bar's OHLC
        htf_idx = i // 4  # Each 1h bar is 1/4 of a 4h bar
        if htf_idx < 1:  # Need at least one previous 4h bar
            signals[i] = 0.0
            continue
            
        # Get previous 4h bar's OHLC (completed bar)
        prev_4h_idx = htf_idx - 1
        if prev_4h_idx >= len(df_4h):
            signals[i] = 0.0
            continue
            
        ph = high_4h[prev_4h_idx]  # previous 4h high
        pl = low_4h[prev_4h_idx]   # previous 4h low
        pc = close_4h[prev_4h_idx] # previous 4h close
        
        # Calculate Camarilla levels
        range_4h = ph - pl
        if range_4h <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        h3 = pc + (range_4h * 1.1 / 4)
        l3 = pc - (range_4h * 1.1 / 4)
        h4 = pc + (range_4h * 1.1 / 2)
        l4 = pc - (range_4h * 1.1 / 2)
        
        # Get current 1h close
        close_price = close_1h[i]
        
        # Get current 1h volume for confirmation
        volume_confirmation = volume_1h[i] > 1.5 * volume_ma_20_1h[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(open_time[i]).hour
        in_session = (8 <= hour <= 20)
        
        if position == 0:  # Flat - look for new entries
            if in_session and volume_confirmation:
                # Long entry: price breaks above H3 + EMA rising
                if (close_price > h3 and ema_rising[i]):
                    position = 1
                    entry_price = close_price
                    signals[i] = 0.20
                # Short entry: price breaks below L3 + EMA falling
                elif (close_price < l3 and ema_falling[i]):
                    position = -1
                    entry_price = close_price
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_1h[i]
                # Exit conditions: price < L3 OR stoploss hit
                if close_price < l3 or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_1h[i]
                # Exit conditions: price > H3 OR stoploss hit
                if close_price > h3 or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals