#!/usr/bin/env python3
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
    
    # === 4h data (Primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA50 for trend direction
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d data (HTF for regime) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA200 for long-term trend filter
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Donchian channels on 1d data
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_upper_1d, donch_lower_1d = donchian_channels(high_1d, low_1d, 20)
    donch_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    
    # === 4h indicators for entry timing ===
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(donch_upper_1d_aligned[i]) or np.isnan(donch_lower_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        ema_200_1d_val = ema_200_1d_aligned[i]
        donch_upper_1d_val = donch_upper_1d_aligned[i]
        donch_lower_1d_val = donch_lower_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 1d Donchian lower OR RSI becomes overbought
            if (price < donch_lower_1d_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 1d Donchian upper OR RSI becomes oversold
            if (price > donch_upper_1d_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price above 1d EMA200 (long-term uptrend) AND price breaks above 1d Donchian upper
                # AND RSI not overbought AND volume spike
                if (price > ema_200_1d_val) and (price > donch_upper_1d_val) and (rsi_val < 60) and (vol_ratio_val > 1.8):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                
                # SHORT: Price below 1d EMA200 (long-term downtrend) AND price breaks below 1d Donchian lower
                # AND RSI not oversold AND volume spike
                elif (price < ema_200_1d_val) and (price < donch_lower_1d_val) and (rsi_val > 40) and (vol_ratio_val > 1.8):
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_EMA_RSI_Volume_Session"
timeframe = "4h"
leverage = 1.0