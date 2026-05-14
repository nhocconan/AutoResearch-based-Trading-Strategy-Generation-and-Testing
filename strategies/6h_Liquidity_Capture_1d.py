#!/usr/bin/env python3
name = "6h_Liquidity_Capture_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for liquidity zones and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Liquidity zones: Previous day's high/low + overnight gap fill zones
    liq_high = prev_high  # Previous day high (liquidity pool)
    liq_low = prev_low    # Previous day low (liquidity pool)
    
    # Overnight gap: today's open vs yesterday's close
    open_1d = df_1d['open'].values
    gap_up = (open_1d > prev_close)  # Gapped up from prev close
    gap_down = (open_1d < prev_close)  # Gapped down from prev close
    
    # Gap fill zones act as liquidity magnets
    liq_gap_up = prev_close  # Gap up tends to fill down to prev close
    liq_gap_down = prev_close  # Gap down tends to fill up to prev close
    
    # Align liquidity zones to 6h timeframe
    liq_high_aligned = align_htf_to_ltf(prices, df_1d, liq_high)
    liq_low_aligned = align_htf_to_ltf(prices, df_1d, liq_low)
    liq_gap_up_aligned = align_htf_to_ltf(prices, df_1d, liq_gap_up)
    liq_gap_down_aligned = align_htf_to_ltf(prices, df_1d, liq_gap_down)
    
    # 1D EMA34 for trend filter (responsive but smooth)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(liq_high_aligned[i]) or np.isnan(liq_low_aligned[i]) or 
            np.isnan(liq_gap_up_aligned[i]) or np.isnan(liq_gap_down_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: Price breaks above liquidity high (prev day high or gap down fill) with volume
            # AND price is above daily EMA34 (bullish bias)
            liq_resistance = max(liq_high_aligned[i], liq_gap_down_aligned[i])
            if (close[i] > liq_resistance and 
                volume_surge and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below liquidity low (prev day low or gap up fill) with volume
            # AND price is below daily EMA34 (bearish bias)
            elif (close[i] < liq_low_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Dynamic exit: price returns to liquidity zone or trend fails
            if position == 1:
                # Exit long: price returns to liquidity support or trend turns bearish
                liq_support = min(liq_low_aligned[i], liq_gap_up_aligned[i])
                if (close[i] < liq_support) or (close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to liquidity resistance or trend turns bullish
                liq_resistance = max(liq_high_aligned[i], liq_gap_down_aligned[i])
                if (close[i] > liq_resistance) or (close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals