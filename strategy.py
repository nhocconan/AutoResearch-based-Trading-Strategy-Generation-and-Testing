# 1. Hypothesis
# In sideways and trending markets, price often reverts to the VWAP (Volume Weighted Average Price) after
# deviating beyond a volatility threshold. This strategy uses:
#   - 4h VWAP as the mean reversion target
#   - 4h Bollinger Bands (20, 2) to detect overextension
#   - 1-day ADX to filter only trending regimes (ADX > 25) where mean reversion is weaker
#   - Volume confirmation: current volume > 20-period average to avoid low-liquidity false signals
# Entry: Long when price < lower BB and VWAP rising; Short when price > upper BB and VWAP falling.
# Exit: When price crosses back to VWAP or regime shifts to range (ADX < 20).
# This combines mean reversion with trend filtering to work in both bull and bear markets.
# Position size: 0.25 (25%) to balance risk and reward, using discrete levels to minimize fee churn.

# 2. Implementation
#!/usr/bin/env python3
name = "4h_VWAP_MeanReversion_ADXFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP (typical price * volume) cumulative, reset daily
    typical_price = (high + low + close) / 3.0
    vwap_raw = typical_price * volume
    # Daily reset: assume index is DatetimeIndex with daily frequency
    # Use pandas Series for resetting by date
    vwap_series = pd.Series(vwap_raw)
    vol_series = pd.Series(volume)
    # Group by date to reset cumulative each day
    dates = pd.to_datetime(prices['open_time']).date
    vwap_cum = vwap_series.groupby(dates).cumsum()
    vol_cum = vol_series.groupby(dates).cumsum()
    vwap = (vwap_cum / vol_cum).values
    # Replace inf/NaN from zero volume days with close price
    vwap = np.where(vol_cum == 0, close, vwap)
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_mid + 2 * bb_std).values
    bb_lower = (bb_mid - 2 * bb_std).values
    bb_mid = bb_mid.values
    
    # 1-day ADX for trend filtering
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # positive when down
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/14)
    tr_1d_series = pd.Series(tr_1d)
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    atr_1d = tr_1d_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * (plus_dm_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10))
    minus_di_1d = 100 * (minus_dm_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10))
    
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe (wait for daily close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > volume_ma20
    
    # VWAP slope: rising if today's VWAP > yesterday's
    vwap_rising = vwap > np.roll(vwap, 1)
    vwap_falling = vwap < np.roll(vwap, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # BB and ADX need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if np.isnan(vwap[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price below lower BB, VWAP rising, trending market (ADX > 25), volume confirmation
            if close[i] < bb_lower[i] and vwap_rising[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price above upper BB, VWAP falling, trending market (ADX > 25), volume confirmation
            elif close[i] > bb_upper[i] and vwap_falling[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above VWAP OR market becomes ranging (ADX < 20)
            if close[i] > vwap[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below VWAP OR market becomes ranging (ADX < 20)
            if close[i] < vwap[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals