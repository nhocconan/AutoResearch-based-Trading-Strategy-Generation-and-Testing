#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume spike
# Camarilla levels provide precise intraday support/resistance. Breakouts above R3 or below S3
# with 1d EMA(34) trend alignment and volume confirmation capture strong momentum moves.
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Works in both bull (breakouts with trend) and bear (failed breakouts reverse) markets.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 1d EMA to 12h (changes only when 1d bar closes)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_1d_series = pd.Series(typical_price_1d.values)
    
    # Camarilla width = (high - low) * 1.1 / 12
    camarilla_width_1d = (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_width_1d_series = pd.Series(camarilla_width_1d.values)
    
    # R3 = typical_price + camarilla_width * 1.1
    # S3 = typical_price - camarilla_width * 1.1
    r3_1d = typical_price_1d_series + camarilla_width_1d_series * 1.1
    s3_1d = typical_price_1d_series - camarilla_width_1d_series * 1.1
    
    # Align Camarilla levels to 12h (use previous day's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20, 14)  # 1d EMA(34), volume MA(20), ATR(14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > R3, above 1d EMA34, volume confirm
            if price > r3_1d_aligned[i] and price > ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Price < S3, below 1d EMA34, volume confirm
            elif price < s3_1d_aligned[i] and price < ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or retracement to 1d EMA34
            # ATR-based stoploss: 2.5 * ATR below entry (wider for 12h)
            stop_loss = entry_price - 2.5 * atr[i]
            if price < stop_loss or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or retracement to 1d EMA34
            # ATR-based stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * atr[i]
            if price > stop_loss or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals