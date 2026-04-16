#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA(50) as trend filter and 1d Donchian(20) breakout with volume confirmation.
# Long when 1w EMA50 is bullish (price > EMA) AND price breaks above 1d Donchian upper band with volume spike (>1.8x median volume).
# Short when 1w EMA50 is bearish (price < EMA) AND price breaks below 1d Donchian lower band with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.0*ATR,
# short exits when price > lowest low since entry + 2.0*ATR.
# Uses discrete position size 0.25. 1w EMA50 identifies primary trend direction to avoid counter-trend trades,
# Donchian breakout captures momentum in the direction of the trend, volume confirmation reduces false signals.
# Designed to work in both bull and bear markets by following the primary trend with precise entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) ===
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if price > EMA50, bearish if price < EMA50
    bullish_trend = close_1w > ema_50
    bearish_trend = close_1w < ema_50
    
    # Get 1d data for Donchian, volume, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian(20), Volume Median, ATR(10) ===
    # Donchian Channels (20-period)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume median (20-period)
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # ATR(10) for trailing stop
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (1d)
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 10)  # EMA(50), Donchian(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        bullish = bullish_trend_aligned[i] > 0.5
        bearish = bearish_trend_aligned[i] > 0.5
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.8x median volume
        volume_spike = current_vol_1d > (vol_median * 1.8)
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 2.0*ATR
            if price < highest_since_entry - 2.0 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 2.0*ATR
            if price > lowest_since_entry + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # 1w EMA50 bullish, price breaks above Donchian upper, volume spike
            if bullish and price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # 1w EMA50 bearish, price breaks below Donchian lower, volume spike
            elif bearish and price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wEMA50_1dDonchian20_Breakout_VolumeSpike1.8x_ATRTrail2.0_v1"
timeframe = "1d"
leverage = 1.0