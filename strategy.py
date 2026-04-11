#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX trend filter
# - Long: price breaks above H3 level with volume > 1.5x 20-period average and 1d ADX > 25
# - Short: price breaks below L3 level with volume > 1.5x 20-period average and 1d ADX > 25
# - Exit: price returns to pivot point (PP) or ATR-based stop (2.0 ATR)
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide precise intraday support/resistance levels
# - Volume confirmation filters breakouts with institutional participation
# - 1d ADX > 25 ensures we only trade when there is sufficient trend strength

name = "1h_4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return signals
    
    # Pre-compute 4h volume SMA(20)
    volume_4h = df_4h['volume'].values
    volume_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivots and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Camarilla pivots (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots using previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current values as fallback
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    range_1d = prev_high - prev_low
    h3 = prev_close + range_1d * 1.1 / 4
    l3 = prev_close - range_1d * 1.1 / 4
    pp = (prev_high + prev_low + prev_close) / 3  # pivot point
    
    # Align 1d Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Pre-compute 1d ADX(14) for trend filter
    # True Range
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute ATR for stoploss (1h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(volume_sma_20_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 4h 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_4h_aligned[i]
        
        # Trend filter: 1d ADX > 25 (indicates sufficient trend strength)
        adx_trend = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above H3 level
        if close_price > h3_aligned[i] and vol_confirm and adx_trend:
            enter_long = True
        
        # Short breakdown: price breaks below L3 level
        if close_price < l3_aligned[i] and vol_confirm and adx_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or ATR-based stop
            exit_long = (close_price <= pp_aligned[i]) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to pivot point or ATR-based stop
            exit_short = (close_price >= pp_aligned[i]) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals