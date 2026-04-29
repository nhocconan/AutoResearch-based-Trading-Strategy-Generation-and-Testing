#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d EMA50 trend filter and volume confirmation
# Bollinger Band squeeze (low volatility contraction) precedes explosive breakouts
# 1d EMA50 provides strong HTF trend filter to align with primary trend direction
# Volume spike (2.0x 20-period average) confirms breakout validity
# ATR-based trailing stop (2.0x ATR) manages risk while allowing trends to develop
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag
# Works in bull markets via long signals when BB squeeze breaks upward and price > 1d EMA50
# Works in bear markets via short signals when BB squeeze breaks downward and price < 1d EMA50
# Bollinger Band squeeze is a proven volatility-based edge that works across market regimes

name = "4h_BollingerSqueeze_EMA50_VolumeConfirm_v1"
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
    upper_band = sma_bb + (bb_std * std_bb)
    lower_band = sma_bb - (bb_std * std_bb)
    bb_width = (upper_band - lower_band) / sma_bb  # Normalized band width
    
    # Bollinger Band squeeze: band width below 20-period percentile (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_std = pd.Series(bb_width).rolling(window=20, min_periods=20).std().values
    squeeze_threshold = bb_width_ma - (1.5 * bb_width_std)  # 1.5 std below mean width
    bb_squeeze = bb_width < squeeze_threshold
    
    # Breakout detection: price breaks above upper band OR below lower band
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Volume spike confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
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
    
    start_idx = 50  # warmup for EMA50 and BB
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_bb_squeeze = bb_squeeze[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_vol_spike = vol_spike[i]
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR breakout fails (price < middle band)
            middle_band = sma_bb[i]
            if curr_close < stop_price or curr_close < middle_band:
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
            # Exit conditions: price above trailing stop OR breakout fails (price > middle band)
            middle_band = sma_bb[i]
            if curr_close > stop_price or curr_close > middle_band:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: BB squeeze breakout upward AND price > 1d EMA50 AND volume spike
            if curr_bb_squeeze and curr_breakout_up and curr_close > curr_ema_1d and curr_vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
            # Short entry: BB squeeze breakout downward AND price < 1d EMA50 AND volume spike
            elif curr_bb_squeeze and curr_breakout_down and curr_close < curr_ema_1d and curr_vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals