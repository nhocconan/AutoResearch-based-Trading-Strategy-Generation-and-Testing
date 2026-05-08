#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d ADX trend strength + volume confirmation
# Uses Choppiness Index to detect ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In ranging markets: mean-reversion at Bollinger Bands (20,2) with RSI confirmation.
# In trending markets: follow 1d ADX trend direction (ADX > 25) with EMA(21) pullback entries.
# Volume spike required for all entries to avoid low-liquidity false signals.
# Designed for low trade frequency in both bull (trending) and bear (ranging) markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "4h_ChopRegime_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) >= period:
            smoothed[period-1] = np.nansum(data[1:period])
            for i in range(period, len(data)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + data[i]
        return smoothed
    
    tr_smoothed = wilders_smooth(tr, 14)
    dm_plus_smoothed = wilders_smooth(dm_plus, 14)
    dm_minus_smoothed = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    adx_1d = wilders_smooth(adx, 14)  # Second smoothing for ADX
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h indicators
    # Choppiness Index (14)
    def choppiness_index(high, low, close, period):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        atr_sum = np.nansum(atr[1:period+1]) if len(atr) >= period+1 else np.nan
        atr_sum_smoothed = np.full_like(atr, np.nan)
        if len(atr) >= period+1:
            atr_sum_smoothed[period] = atr_sum
            for i in range(period+1, len(atr)):
                atr_sum_smoothed[i] = atr_sum_smoothed[i-1] - (atr_sum_smoothed[i-1]/period) + atr[i]
        
        max_high = np.maximum.accumulate(high)
        min_low = np.minimum.accumulate(low)
        range_max_min = max_high - min_low
        
        chop = np.full_like(close, np.nan)
        valid = (range_max_min != 0) & (~np.isnan(atr_sum_smoothed))
        chop[valid] = 100 * np.log10(atr_sum_smoothed[valid] / range_max_min[valid]) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + bb_std_dev * bb_std
    bb_lower = bb_ma - bb_std_dev * bb_std
    
    # RSI(14)
    def rsi(close, period):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        if len(close) >= period+1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # EMA(21) for pullback entries in trending markets
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(bb_ma[i]) or np.isnan(rsi_vals[i]) or
            np.isnan(ema21[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        chop_val = chop[i]
        bb_ma_val = bb_ma[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        rsi_val = rsi_vals[i]
        ema21_val = ema21[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        if position == 0:
            # Ranging market: CHOP > 61.8
            if chop_val > 61.8:
                # Mean reversion at Bollinger Bands with RSI confirmation
                if close_val <= bb_lower_val and rsi_val < 30 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close_val >= bb_upper_val and rsi_val > 70 and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Trending market: CHOP < 38.2 and ADX > 25
            elif chop_val < 38.2 and adx_val > 25:
                # Trend direction from 1d ADX components (need DI+ and DI-)
                # We'll use close vs EMA21 as proxy for trend direction simplicity
                if close_val > ema21_val and vol_spike:
                    # Pullback long in uptrend
                    if close_val <= ema21_val * 1.02:  # Within 2% of EMA21
                        signals[i] = 0.25
                        position = 1
                elif close_val < ema21_val and vol_spike:
                    # Pullback short in downtrend
                    if close_val >= ema21_val * 0.98:  # Within 2% of EMA21
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: reverse signal or volatility expansion
            if chop_val > 61.8 and close_val >= bb_ma_val:  # Return to mean in ranging
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and adx_val > 25 and close_val < ema21_val:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: reverse signal or volatility expansion
            if chop_val > 61.8 and close_val <= bb_ma_val:  # Return to mean in ranging
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and adx_val > 25 and close_val > ema21_val:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals