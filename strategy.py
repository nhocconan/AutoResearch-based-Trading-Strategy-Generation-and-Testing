#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze breakout with 1d EMA50 trend filter and volume confirmation
# Identifies low volatility periods (BB Width at 20-period low) followed by breakouts in direction of 1d EMA50
# Volume spike (>2.0x 20-period average) confirms institutional participation
# ATR trailing stop (2.0x ATR) manages risk while allowing trends to develop
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag
# Works in bull markets via long breakouts above upper BB when price > 1d EMA50
# Works in bear markets via short breakouts below lower BB when price < 1d EMA50
# Bollinger Squeeze is effective in ranging markets by avoiding false signals during low volatility

name = "4h_BollingerSqueeze_EMA50_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + bb_std * std_bb
    lower_bb = sma_bb - bb_std * std_bb
    bb_width = upper_bb - lower_bb
    
    # Calculate BB Width percentile (50-period lookback) for squeeze detection
    bb_width_percentile = np.full_like(bb_width, np.nan, dtype=float)
    for i in range(bb_period + 49, n):  # Need 50 bb_width values + bb_period for SMA
        start_idx = max(0, i - 49)
        bb_width_window = bb_width[start_idx:i+1]
        if len(bb_width_window) >= 10:  # Minimum window for percentile
            current_width = bb_width[i]
            if not np.isnan(current_width):
                percentile = (np.sum(bb_width_window <= current_width) / len(bb_width_window)) * 100
                bb_width_percentile[i] = percentile
    
    # Bollinger Squeeze: BB Width at or below 20th percentile (low volatility)
    squeeze_condition = bb_width_percentile <= 20.0
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = bb_period + 49  # warmup for BB and percentile
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_sma_bb = sma_bb[i]
        curr_upper_bb = upper_bb[i]
        curr_lower_bb = lower_bb[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_squeeze = squeeze_condition[i] if not np.isnan(squeeze_condition[i]) else False
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR breakout fails (price < middle BB)
            if curr_close < stop_price or curr_close < curr_sma_bb:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR breakout fails (price > middle BB)
            if curr_close > stop_price or curr_close > curr_sma_bb:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: BB squeeze breakout above upper BB AND price > 1d EMA50 AND volume spike
            if curr_close > curr_upper_bb and curr_close > curr_ema_1d and vol_spike and not curr_squeeze:
                # Ensure we were in squeeze recently (within last 5 bars)
                was_squeezed = False
                for j in range(max(0, i-5), i):
                    if j < len(squeeze_condition) and squeeze_condition[j]:
                        was_squeezed = True
                        break
                if was_squeezed:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high_since_entry = curr_high
            # Short entry: BB squeeze breakout below lower BB AND price < 1d EMA50 AND volume spike
            elif curr_close < curr_lower_bb and curr_close < curr_ema_1d and vol_spike and not curr_squeeze:
                # Ensure we were in squeeze recently (within last 5 bars)
                was_squeezed = False
                for j in range(max(0, i-5), i):
                    if j < len(squeeze_condition) and squeeze_condition[j]:
                        was_squeezed = True
                        break
                if was_squeezed:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals