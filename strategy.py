#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and ADX regime filter
# Long when price breaks above Camarilla R1 level + 1d volume > 2.0x 20-period avg + 1d ADX > 25 (trending)
# Short when price breaks below Camarilla S1 level + 1d volume > 2.0x 20-period avg + 1d ADX > 25 (trending)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla pivot levels provide high-probability reversal/breakout points derived from prior day's range.
# Volume confirmation ensures breakouts have conviction. ADX filter avoids ranging markets where false breakouts occur.
# Target: 20-35 trades/year on 12h timeframe to stay within fee drag limits.

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
    
    # === 1d Indicators: Volume SMA and ADX ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # Volume SMA (20-period)
    vol_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # ADX (14-period) - requires +DI, -DI, DX calculation
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
                else:
                    result[i] = np.nan
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 12h Camarilla Pivot Levels (R1, S1) ===
    # Based on prior 12h bar's high, low, close
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # We need to shift by 1 to use prior bar's data (no look-ahead)
    shift_high = np.concatenate([[np.nan], high[:-1]])
    shift_low = np.concatenate([[np.nan], low[:-1]])
    shift_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_r1 = shift_close + (shift_high - shift_low) * 1.1 / 12
    camarilla_s1 = shift_close - (shift_high - shift_low) * 1.1 / 12
    
    # Volume SMA for 12h confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # 1d indicators(50) + Camarilla(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filters: current 12h volume > 2.0x 20-period volume SMA
        # AND 1d volume > 2.0x 20-period volume SMA (HTF confirmation)
        vol_confirm_12h = volume[i] > (vol_sma_20[i] * 2.0)
        vol_confirm_1d = vol_1d[i] > (vol_sma_20_1d_aligned[i] * 2.0) if i < len(vol_1d) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # Regime filter: 1d ADX > 25 (trending market)
        regime_filter = adx_1d_aligned[i] > 25.0
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 level (close > R1)
        # 2. Volume confirmation (both timeframes)
        # 3. Trending regime (ADX > 25)
        if (close[i] > camarilla_r1[i]) and vol_confirm and regime_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 level (close < S1)
        # 2. Volume confirmation (both timeframes)
        # 3. Trending regime (ADX > 25)
        elif (close[i] < camarilla_s1[i]) and vol_confirm and regime_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dVolume_ADX_Filter_v1"
timeframe = "12h"
leverage = 1.0