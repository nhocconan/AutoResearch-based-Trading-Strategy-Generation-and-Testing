#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H4/L4 breakout with 1w EMA34 trend filter and volume spike confirmation.
# Camarilla pivot levels provide strong intraday support/resistance that often holds in ranging markets
# and breaks with conviction in trending markets. 1w EMA34 ensures we trade with the weekly trend.
# Volume spike (>2.0x 20-bar average) confirms breakout significance. Works in both bull (long above EMA34)
# and bear (short below EMA34) regimes by filtering breakout direction. Target: 30-100 trades over 4 years (7-25/year).
# Size: 0.25.

name = "1d_Camarilla_H4L4_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla levels (H4, L4) from previous day
    # H4 = Close + 1.1/2 * (High - Low)
    # L4 = Close - 1.1/2 * (High - Low)
    # Using previous day's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close as previous (no prior data)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_h4 = prev_close + 1.1/2 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1/2 * (prev_high - prev_low)
    
    # 1d ATR(14) for volume confirmation context (not used directly)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 bars for volume MA and 1 for previous day
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA34 direction
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Breakout conditions at Camarilla H4/L4 levels
        long_breakout = close[i] > camarilla_h4[i]
        short_breakout = close[i] < camarilla_l4[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit: opposite Camarilla level touch (H4 for long exit, L4 for short exit)
        long_exit = close[i] < camarilla_h4[i]
        short_exit = close[i] > camarilla_l4[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals