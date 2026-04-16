#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(21) pullback strategy with 4h Donchian(20) trend filter and 1d volume confirmation.
# Long when: 1h price > EMA21 AND 1h close > 1h open (bullish candle) AND price > 4h Donchian upper(20) AND 1d volume > 1.3x 20-period average
# Short when: 1h price < EMA21 AND 1h close < 1h open (bearish candle) AND price < 4h Donchian lower(20) AND 1d volume > 1.3x 20-period average
# Exit when price crosses EMA21 in opposite direction.
# Uses 4h/1d for signal direction and regime, 1h for precise entry timing.
# Session filter: 08-20 UTC to avoid low-volume Asian session.
# Position size: 0.20 discrete levels to minimize fee churn.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === 4h Indicators: Donchian channels (20-period) ===
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # === 1h Indicators: EMA(21) for dynamic support/resistance ===
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_21[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_val = ema_21[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        is_bullish_candle = close[i] > open_price[i]
        is_bearish_candle = close[i] < open_price[i]
        
        # Volume filter: volume > 1.3x 20-period average (using 1d volume MA)
        vol_filter = vol > 1.3 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below EMA21
            if price < ema_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above EMA21
            if price > ema_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price above EMA21, bullish candle, above 4h Donchian upper, with volume confirmation
            if price > ema_val and is_bullish_candle and price > upper_val and vol_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: price below EMA21, bearish candle, below 4h Donchian lower, with volume confirmation
            elif price < ema_val and is_bearish_candle and price < lower_val and vol_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA21_Pullback_4hDonchian20_1dVolumeFilter_V1"
timeframe = "1h"
leverage = 1.0