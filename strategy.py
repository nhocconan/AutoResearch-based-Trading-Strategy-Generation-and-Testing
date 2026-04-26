#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime_v4
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakout in the direction of 1d EMA34 trend with volume spike (>1.5x 20-period MA) and choppiness regime filter (CHOP > 61.8 for ranging market) captures high-probability mean-reversion bounces in ranging markets and breakouts in trending markets. Uses discrete position sizing (±0.30) and ATR-based trailing stop (2.0x). Targets 20-50 trades/year by requiring confluence of pivot breakout, trend alignment, volume confirmation, and regime filter—designed to work in both bull (buy dips to R1 in uptrend) and bear (sell rallies to S1 in downtrend) markets with BTC/ETH edge from institutional pivot levels and volume-confirmed mean reversion.
"""

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
    
    # Load 1d data ONCE before loop for Camarilla pivot, EMA34 trend, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla pivot calculation (using previous day's close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close (for Camarilla calculation)
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan  # First value has no previous
    
    # Camarilla pivot levels: based on previous day's range
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_1d - low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12.0
    s1 = prev_close_1d - 1.1 * camarilla_range / 12.0
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    close_series = pd.Series(close_1d)
    ema34_1d = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d ATR(14) for trailing stop
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h volume spike filter: volume > 1.5 * 20-period MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # 4h choppiness regime filter: CHOP(14) > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    def calculate_choppiness(high, low, close, period=14):
        atr_sum = np.zeros_like(close)
        true_range = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            true_range[i] = tr
            if i >= period:
                atr_sum[i] = atr_sum[i-1] + true_range[i] - true_range[i-period+1]
            else:
                atr_sum[i] = atr_sum[i-1] + true_range[i]
        atr_period = np.where(np.arange(len(close)) >= period-1, atr_sum[period-1:] / period, np.nan)
        atr_period_full = np.full_like(close, np.nan)
        atr_period_full[period-1:] = atr_period
        max_high = np.full_like(close, np.nan)
        min_low = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i >= period-1:
                max_high[i] = np.max(high[i-period+1:i+1])
                min_low[i] = np.min(low[i-period+1:i+1])
        chop = np.where((max_high - min_low) > 0, 100 * np.log10(atr_period_full / (max_high - min_low)) / np.log10(period), 50)
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    chop_regime_ranging = chop > 61.8  # Ranging market: mean revert at pivot levels
    chop_regime_trending = chop < 38.2  # Trending market: follow EMA34 trend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Camarilla (needs 1d), EMA34 (34), ATR (14), volume MA (20), chop (14)
    start_idx = max(1, 34, 14, 20, 14) + 28  # +28 to ensure 1 day of 4h data for daily indicators
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34_val = ema34_aligned[i]
        atr_val = atr_aligned[i]
        vol_spike = volume_spike[i]
        chop_ranging = chop_regime_ranging[i]
        chop_trending = chop_regime_trending[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema34_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Entry conditions: Camarilla breakout with confluence
        # Long: price breaks above R1 + volume spike + (ranging market OR trending market with price > EMA34)
        long_breakout = close_val > r1_val
        long_condition = long_breakout and vol_spike and (chop_ranging or (chop_trending and close_val > ema34_val))
        
        # Short: price breaks below S1 + volume spike + (ranging market OR trending market with price < EMA34)
        short_breakout = close_val < s1_val
        short_condition = short_breakout and vol_spike and (chop_ranging or (chop_trending and close_val < ema34_val))
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss (2.0x ATR)
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime_v4"
timeframe = "4h"
leverage = 1.0