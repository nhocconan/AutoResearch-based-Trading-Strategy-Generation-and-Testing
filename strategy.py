#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and session filter (08-20 UTC).
# Uses 4h EMA50 for trend alignment (long only in uptrend, short only in downtrend) to avoid counter-trend trades.
# Enters on 1h break of Camarilla R3 (long) or S3 (short) with volume > 1.5x 20-period MA for confirmation.
# Exits on opposite Camarilla level (R3 for shorts, S3 for longs) or trend reversal.
# Discrete sizing 0.20. Session filter reduces noise trades outside active hours.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Camarilla levels provide precise intraday support/resistance; 4h EMA50 filters regime; volume confirms breakout.

name = "1h_Camarilla_R3S3_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss and volatility
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from prior 1h bar (H1, L1, C1)
    # Need to shift by 1 to use completed 1h bar only (no look-ahead)
    h1 = pd.Series(high).shift(1).values
    l1 = pd.Series(low).shift(1).values
    c1 = pd.Series(close).shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = c1 + (h1 - l1) * 1.1 / 4
    camarilla_s3 = c1 - (h1 - l1) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1h bar)
    r3_level = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), camarilla_r3)
    s3_level = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), camarilla_s3)
    
    # Volume regime: current 1h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_level[i]) or 
            np.isnan(s3_level[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r3 = r3_level[i]
        s3 = s3_level[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above R3 with volume spike in uptrend
            if close_val > r3 and vol_spike and is_uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            # Short: break below S3 with volume spike in downtrend
            elif close_val < s3 and vol_spike and is_downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: ATR-based stoploss OR price breaks below S3 OR trend turns down
            if close_val < entry_price - 2.0 * atr_val or close_val < s3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: ATR-based stoploss OR price breaks above R3 OR trend turns up
            if close_val > entry_price + 2.0 * atr_val or close_val > r3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals