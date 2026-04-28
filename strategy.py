#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# Uses Bollinger Bands calculated from 6h data to detect low volatility squeeze (BB width < 20th percentile)
# Breakout occurs when price closes outside BB AND 1w EMA50 trend aligned AND volume > 2.0x average
# Designed to work in both bull and bear markets by combining volatility contraction with trend following
# Target: 12-37 trades/year via tight Bollinger squeeze breakout conditions + volume + trend filter

name = "6h_Bollinger_Squeeze_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h data
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width < 20th percentile (low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe (completed 1w candles only)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # Need sufficient history for BB percentile and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_squeeze[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        ema50_val = ema50_1w_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price closes above upper BB AND in squeeze AND 1w EMA50 uptrend AND volume spike
            if price > bb_up and bb_squeeze[i] and price > ema50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price closes below lower BB AND in squeeze AND 1w EMA50 downtrend AND volume spike
            elif price < bb_low and bb_squeeze[i] and price < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below middle BB (mean reversion)
            # ATR-based stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            # Exit on stoploss or price < middle BB (mean reversion)
            if price < stop_loss or price < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or price rises above middle BB (mean reversion)
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            # Exit on stoploss or price > middle BB (mean reversion)
            if price > stop_loss or price > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals