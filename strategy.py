#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX-based trend strength with volume confirmation and daily trend filter.
# Uses ADX (14) to identify trending markets (ADX > 25) and avoids ranging markets.
# Daily EMA50 provides higher timeframe trend bias for directional filtering.
# Volume confirmation (current volume > 1.3x 20-period average) ensures institutional participation.
# Designed for 6h timeframe to target 50-150 trades over 4 years.
# Works in bull/bear markets via ADX trend filter and daily EMA bias - avoids false signals in ranging markets.

name = "6h_adx_volume_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily closes
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    
    # Align EMA50 to 6h timeframe (shifted by 1 daily bar for no look-ahead)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX calculation (14 periods)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: smoothed = (prev_smooth * (period-1) + current) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smoothed = wilders_smooth(tr, period)
    plus_dm_smoothed = wilders_smooth(plus_dm, period)
    minus_dm_smoothed = wilders_smooth(minus_dm, period)
    
    # Directional Indicators
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(period, n):
        if not np.isnan(tr_smoothed[i]) and tr_smoothed[i] != 0:
            plus_di[i] = (plus_dm_smoothed[i] / tr_smoothed[i]) * 100
            minus_di[i] = (minus_dm_smoothed[i] / tr_smoothed[i]) * 100
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX (smoothed DX)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 6h timeframe (no additional shift needed as Wilder's smoothing already uses past data)
    adx_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(adx))}), adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after ADX is available (2*period for smoothing)
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        if np.isnan(vol_ma):
            volume_filter = False
        else:
            volume_filter = volume[i] > vol_ma * 1.3
        
        # Trend strength: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Directional bias from DI crossover
        bullish_momentum = plus_di[i] > minus_di[i]
        bearish_momentum = minus_di[i] > plus_di[i]
        
        # Trend bias: daily EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: loss of trend strength or stoploss (2x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (not trending or 
                not bullish_momentum or
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: loss of trend strength or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (not trending or 
                not bearish_momentum or
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of daily trend with volume confirmation
            if volume_filter and trending:
                # Long: bullish momentum in uptrend
                if bullish_momentum and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bearish momentum in downtrend
                elif bearish_momentum and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals