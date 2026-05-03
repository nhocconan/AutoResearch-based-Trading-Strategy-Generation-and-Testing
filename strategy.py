#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when: price breaks above 12h Camarilla R3 level AND close > 1d EMA34 AND volume > 2.0x 24-bar average
# Short when: price breaks below 12h Camarilla S3 level AND close < 1d EMA34 AND volume > 2.0x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 12h Camarilla for structure (proven edge from top performers), 1d EMA34 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Camarilla_R3_S3_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla pivots (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar (index -2) for Camarilla calculation
    ph = high_12h[-2]  # previous 12h high
    pl = low_12h[-2]   # previous 12h low
    pc = close_12h[-2] # previous 12h close
    
    # Camarilla levels
    R3 = pc + (ph - pl) * 1.1 / 4
    S3 = pc - (ph - pl) * 1.1 / 4
    
    # Align 12h Camarilla levels to 12h timeframe (no additional delay needed as they're based on completed bar)
    R3_12h = align_htf_to_ltf(prices, df_12h, np.full_like(close_12h, R3))
    S3_12h = align_htf_to_ltf(prices, df_12h, np.full_like(close_12h, S3))
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(24) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Start loop from sufficient lookback
    start_idx = max(34, 24)  # EMA34 and ATR24 need min_periods
    
    for i in range(start_idx, n):
        # Get current values
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r3_level = R3_12h[i]
        s3_level = S3_12h[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ok = vol_spike[i]
        atr_val = atr[i]
        
        # Entry logic
        if position == 0:
            # Long: break above R3, above EMA34 trend, volume spike
            if curr_close > r3_level and curr_close > ema_trend and vol_ok:
                position = 1
                signals[i] = 0.25  # 25% position
                entry_price = curr_close
                highest_high = curr_high
                lowest_low = curr_low
            # Short: break below S3, below EMA34 trend, volume spike
            elif curr_close < s3_level and curr_close < ema_trend and vol_ok:
                position = -1
                signals[i] = -0.25  # -25% position
                entry_price = curr_close
                highest_high = curr_high
                lowest_low = curr_low
        
        # Exit logic (trailing stop)
        elif position == 1:  # Long position
            highest_high = max(highest_high, curr_high)
            exit_level = highest_high - 2.5 * atr_val
            if curr_close < exit_level:
                position = 0
                signals[i] = 0.0  # exit
        
        elif position == -1:  # Short position
            lowest_low = min(lowest_low, curr_low)
            exit_level = lowest_low + 2.5 * atr_val
            if curr_close > exit_level:
                position = 0
                signals[i] = 0.0  # exit
    
    return signals