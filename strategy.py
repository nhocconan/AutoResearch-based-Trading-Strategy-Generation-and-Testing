#!/usr/bin/env python3
# 1h_OrderBlock_Filter_4hTrend_1dVolume
# Hypothesis: Combines 4h trend direction (EMA50) with 1h order block imbalances and 1d volume surge confirmation.
# Order blocks identified by strong directional candles followed by consolidation - high probability reversal zones.
# Uses 4h EMA50 for trend filter, 1h for precise entry timing, 1d volume filter to avoid low-conviction moves.
# Target: 15-30 trades/year to minimize fee drag while capturing high-probability setups.

name = "1h_OrderBlock_Filter_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume average (20-period) for conviction filter
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Identify order blocks on 1h timeframe
    # Bullish OB: strong down candle followed by consolidation then break above
    # Bearish OB: strong up candle followed by consolidation then break below
    
    # Calculate candle strength and body size
    body_size = np.abs(close - open_)
    candle_range = high - low
    # Avoid division by zero
    body_ratio = np.divide(body_size, candle_range, out=np.zeros_like(body_size), where=candle_range!=0)
    
    # Identify potential order block candles (strong directional moves)
    strong_down = (close < open_) & (body_ratio > 0.6)  # Strong bearish candle
    strong_up = (close > open_) & (body_ratio > 0.6)    # Strong bullish candle
    
    # Find consolidation after strong moves (3-5 candles with small body)
    def find_consolidation_after_strong(strong_signal, lookback=5, consolidation_bars=3):
        consolidation = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            if strong_signal[i-lookback]:
                # Check if next 3-5 candles are consolidating (small body)
                consol_count = 0
                for j in range(1, min(consolidation_bars+2, n-i)):
                    if i+j < n and body_ratio[i+j] < 0.3:  # Small body candle
                        consol_count += 1
                    else:
                        break
                if consol_count >= consolidation_bars:
                    # Mark the consolidation zone
                    for k in range(consol_count):
                        if i+1+k < n:
                            consolidation[i+1+k] = True
        return consolidation
    
    bullish_ob_zone = find_consolidation_after_strong(strong_down, consolidation_bars=3)
    bearish_ob_zone = find_consolidation_after_strong(strong_up, consolidation_bars=3)
    
    # Breakout detection from order block zones
    bullish_break = bullish_ob_zone & (close > high_roll)  # Break above OB high
    bearish_break = bearish_ob_zone & (close < low_roll)   # Break below OB low
    
    # Rolling high/low for breakout confirmation (10-period)
    high_roll = pd.Series(high).rolling(window=10, min_periods=1).max().values
    low_roll = pd.Series(low).rolling(window=10, min_periods=1).min().values
    
    # Recalculate breaks with proper rolling windows
    bullish_break = bullish_ob_zone & (close > high_roll)
    bearish_break = bearish_ob_zone & (close < low_roll)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for 4h EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish order block break + 4h uptrend + volume surge
            if (bullish_break[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.5 * vol_avg_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: bearish order block break + 4h downtrend + volume surge
            elif (bearish_break[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below order block low or trend reversal
            ob_low = np.where(bullish_ob_zone[:i+1], low, np.nan)
            ob_low_valid = ob_low[~np.isnan(ob_low)]
            if len(ob_low_valid) > 0:
                recent_ob_low = ob_low_valid[-1]  # Most recent OB low
                if close[i] < recent_ob_low or close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:
                # Fallback exit: trend reversal
                if close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above order block high or trend reversal
            ob_high = np.where(bearish_ob_zone[:i+1], high, np.nan)
            ob_high_valid = ob_high[~np.isnan(ob_high)]
            if len(ob_high_valid) > 0:
                recent_ob_high = ob_high_valid[-1]  # Most recent OB high
                if close[i] > recent_ob_high or close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:
                # Fallback exit: trend reversal
                if close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

# Note: This implementation focuses on high-probability order block setups with
# multi-timeframe confirmation to keep trade frequency low (target 15-30/year)
# while maintaining edge in both bull and bear markets through trend filtering.