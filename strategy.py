#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h CRSI (Connors RSI) with 1d ADX trend filter and volume confirmation
    # CRSI = (RSI(3) + RSI of Streak + PercentRank(100)) / 3
    # Identifies extreme short-term momentum reversals.
    # In trending markets (ADX > 25), extreme CRSI readings often precede continuation.
    # In ranging markets (ADX < 20), extreme CRSI often precedes mean reversion.
    # Volume confirmation filters low-conviction moves.
    # Target: 20-40 trades/year per symbol with disciplined entries.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def ma(values, period):
        result = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = ma(tr, 14)
    dm_plus_smooth = ma(dm_plus, 14)
    dm_minus_smooth = ma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = ma(dx, 14)
    adx_14 = adx  # ADX(14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate CRSI components
    # RSI(3)
    def rsi(close_prices, period):
        delta = np.diff(close_prices)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan, dtype=np.float64)
        avg_loss = np.full_like(close_prices, np.nan, dtype=np.float64)
        
        if len(close_prices) >= period + 1:
            avg_gain[period] = np.nanmean(gain[1:period+1])
            avg_loss[period] = np.nanmean(loss[1:period+1])
            
            for i in range(period+1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi3 = rsi(close, 3)
    
    # Streak RSI
    def calculate_streak_rsi(close_prices, rsi_period=2, streak_lookback=2):
        # Calculate streak
        streak = np.zeros(len(close_prices))
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i-1]:
                streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
            elif close_prices[i] < close_prices[i-1]:
                streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
            else:
                streak[i] = 0
        
        # RSI of streak values
        return rsi(streak, rsi_period)
    
    streak_rsi = calculate_streak_rsi(close, 2, 2)
    
    # PercentRank(100) - percentage of days where close was lower over last 100 periods
    def percent_rank(close_prices, lookback):
        pr = np.full_like(close_prices, np.nan, dtype=np.float64)
        for i in range(lookback, len(close_prices)):
            window = close_prices[i-lookback:i+1]
            pr[i] = (np.sum(window < close_prices[i]) / len(window)) * 100
        return pr
    
    percent_rank_100 = percent_rank(close, 100)
    
    # CRSI = (RSI(3) + Streak RSI(2) + PercentRank(100)) / 3
    crsi = (rsi3 + streak_rsi + percent_rank_100) / 3.0
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if data not ready
        if (np.isnan(crsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(rsi3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Oversold CRSI + volume spike + trend filter
            # In trending markets (ADX > 25): look for pullbacks in uptrend
            # In ranging markets (ADX < 20): look for mean reversion from oversold
            if crsi[i] < 15 and vol_spike[i]:
                if adx_aligned[i] > 25:  # Trending market
                    # Only go long if price is above 20-period EMA (uptrend)
                    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                    if close[i] > ema20[i]:
                        signals[i] = 0.25
                        position = 1
                else:  # Ranging market
                    # Mean reversion from oversold
                    signals[i] = 0.25
                    position = 1
            # Short conditions: Overbought CRSI + volume spike + trend filter
            elif crsi[i] > 85 and vol_spike[i]:
                if adx_aligned[i] > 25:  # Trending market
                    # Only go short if price is below 20-period EMA (downtrend)
                    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                    if close[i] < ema20[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging market
                    # Mean reversion from overbought
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: CRSI > 50 (momentum fading) or stop/reversal signal
                if crsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: CRSI < 50 (momentum fading) or stop/reversal signal
                if crsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_CRSI_ADX_TrendFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0