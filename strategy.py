#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (EMA34) and 1d volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 4h EMA34 (bullish trend) AND 1d volume > 1.5x MA20.
# Short when price breaks below S3 AND close < 4h EMA34 (bearish trend) AND 1d volume > 1.5x MA20.
# Exit when price crosses the 4h EMA34 in opposite direction.
# Uses 4h HTF for trend to reduce noise and overtrading, 1d for volume regime filter.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Position size: 0.20 (20% of capital) to manage drawdown in bear markets.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Indicators (LTF) ---
    # Volume confirmation: > 1.5x 20-period average (using 1h data for entry timing)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1h = volume > (1.5 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(34) - trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume MA(20) - regime filter (needs extra delay for confirmation)
    vol_ma_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d, additional_delay_bars=0)
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d_aligned)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d, additional_delay_bars=0)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using previous bar's OHLC)
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels
            R3 = prev_close + range_ * 1.1 / 2
            S3 = prev_close - range_ * 1.1 / 2
        else:
            R3 = np.nan
            S3 = np.nan
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 4h EMA34 (bullish trend) AND 1d volume confirm
            if (not np.isnan(R3) and 
                close[i] > R3 and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_confirm_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 AND close < 4h EMA34 (bearish trend) AND 1d volume confirm
            elif (not np.isnan(S3) and 
                  close[i] < S3 and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_confirm_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 4h EMA34 (trend change)
            if close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above 4h EMA34 (trend change)
            if close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals