#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R4 + 1w EMA34 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below 1d Camarilla S4 + 1w EMA34 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.30) to balance return and drawdown.
# 1w EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-25 trades/year to minimize fee drag on 1d timeframe.
# Camarilla levels calculated from previous day's OHLC using 1d HTF data.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Previous day OHLC for Camarilla levels ===
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    # For each 1d bar, get the previous completed day's OHLC
    for i in range(n):
        current_time = prices['open_time'].iloc[i]
        # Find the 1d bar that completed before current_time
        mask = df_1d['open_time'] < current_time
        if mask.any():
            idx = mask.sum() - 1  # Get the last completed 1d bar
            if idx >= 0:
                prev_day_high[i] = df_1d['high'].iloc[idx]
                prev_day_low[i] = df_1d['low'].iloc[idx]
                prev_day_close[i] = df_1d['close'].iloc[idx]
    
    # Calculate Camarilla levels: R4 = close + ((high - low) * 1.1/2), S4 = close - ((high - low) * 1.1/2)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i])):
            high_low_diff = prev_day_high[i] - prev_day_low[i]
            camarilla_r4[i] = prev_day_close[i] + (high_low_diff * 1.1 / 2)
            camarilla_s4[i] = prev_day_close[i] - (high_low_diff * 1.1 / 2)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w Indicator: EMA34 ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # EMA34 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above camarilla R4 (close > R4)
        # 2. 1w EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4[i]) and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below camarilla S4 (close < S4)
        # 2. 1w EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4[i]) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Camarilla_R4S4_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0