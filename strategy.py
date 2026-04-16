#!/usr/bin/env python3
"""
1D_3ATR_Channel_Breakout_TrendFilter
Hypothesis: On daily timeframe, a 3ATR channel breakout with ADX trend filter captures momentum in both bull and bear markets.
Breakouts above upper channel (mean + 3*ATR) go long, below lower channel (mean - 3*ATR) go short, only when ADX > 25.
Uses weekly timeframe for higher-timeframe trend confirmation to avoid counter-trend trades.
Target: 15-30 trades per year (~60-120 total over 4 years) with disciplined entries.
"""
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
    
    # === Daily ATR (14-period) for channel width ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nanmean(x[1:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1]/period) + x[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    
    # === Daily SMA(20) as mean for channel ===
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # === Upper and Lower Channels (SMA ± 3*ATR) ===
    upper_channel = sma_20 + (3.0 * atr)
    lower_channel = sma_20 - (3.0 * atr)
    
    # === Daily ADX(14) for trend filter ===
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed DM and ATR
    atr_smooth = wilders_smoothing(tr, 14)  # Already calculated above
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_smooth != 0, 100 * dm_plus_smooth / atr_smooth, 0)
    di_minus = np.where(atr_smooth != 0, 100 * dm_minus_smooth / atr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # === Weekly trend filter (higher timeframe) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align daily indicators to daily timeframe (no alignment needed as already daily)
    # But we'll keep the pattern for consistency
    upper_channel_aligned = upper_channel
    lower_channel_aligned = lower_channel
    adx_aligned = adx
    
    signals = np.zeros(n)
    
    # Warmup: enough for ATR, SMA, ADX calculations
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = upper_channel_aligned[i]
        lower = lower_channel_aligned[i]
        adx_val = adx_aligned[i]
        weekly_ema = ema_20_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below SMA(20) OR weekly trend turns bearish
            if price < sma_20[i] or price < weekly_ema:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above SMA(20) OR weekly trend turns bullish
            if price > sma_20[i] or price > weekly_ema:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only take trades when ADX indicates trending market
            if adx_val > 25:
                # LONG: Break above upper channel with weekly uptrend confirmation
                if price > upper and price > weekly_ema:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Break below lower channel with weekly downtrend confirmation
                elif price < lower and price < weekly_ema:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1D_3ATR_Channel_Breakout_TrendFilter"
timeframe = "1d"
leverage = 1.0