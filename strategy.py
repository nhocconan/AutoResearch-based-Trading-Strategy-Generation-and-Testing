#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high_20w = np.full(len(high_1w), np.nan)
    donchian_low_20w = np.full(len(low_1w), np.nan)
    for i in range(19, len(high_1w)):
        donchian_high_20w[i] = np.max(high_1w[i-19:i+1])
        donchian_low_20w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20w)
    donchian_low_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20w)
    
    # Calculate daily ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_di[period] = np.mean(plus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        minus_di[period] = np.mean(minus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_dm[i] / atr[i] * 100) if atr[i] != 0 else 0
            minus_di[i] = (minus_dm[i] / atr[i] * 100) if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) != 0 else 0
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period if i > period else dx[i]
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate weekly RSI (14-period) for overbought/oversold conditions
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        rs = np.zeros_like(close)
        rsi = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
            rs[i] = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs[i])) if rs[i] != 0 else (100 if avg_gain[i] > 0 else 0)
        
        return rsi
    
    rsi_14_1w = calculate_rsi(close_1w, 14)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Volume filter: volume > 2.0 x 20-period average (6h periods)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly Donchian (20), daily ADX (14), weekly RSI (14), volume MA (20)
    start_idx = max(20, 14, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20w_aligned[i]) or np.isnan(donchian_low_20w_aligned[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Weekly Donchian levels
        donchian_high = donchian_high_20w_aligned[i]
        donchian_low = donchian_low_20w_aligned[i]
        
        # Daily ADX trend filter (strong trend > 25)
        strong_trend = adx_14_aligned[i] > 25
        
        # Weekly RSI levels
        rsi_value = rsi_14_aligned[i]
        oversold = rsi_value < 30
        overbought = rsi_value > 70
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + volume + strong trend + not overbought
            if price > donchian_high and vol_filter and strong_trend and not overbought:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian low + volume + strong trend + not oversold
            elif price < donchian_low and vol_filter and strong_trend and not oversold:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low or trend weakens
            if price < donchian_low or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high or trend weakens
            if price > donchian_high or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyDonchian20_ADX14_RSI14_Volume"
timeframe = "6h"
leverage = 1.0