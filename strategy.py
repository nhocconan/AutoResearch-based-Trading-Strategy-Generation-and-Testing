#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# In trending markets (ADX > 25): Buy when Bull Power > 0 and rising, Sell when Bear Power < 0 and falling
# In ranging markets (ADX < 20): Fade extremes (Buy when Bear Power < -std and turning up, Sell when Bull Power > std and turning down)
# Volume confirmation filters low-conviction moves. Target: 15-35 trades/year (60-140 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    def calculate_atr(high, low, close, period):
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = np.zeros_like(tr)
        atr[:period-1] = np.nan
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    def calculate_dx(high, low, close, period):
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        
        atr_tr = np.zeros_like(tr)
        atr_tr[:period-1] = np.nan
        if len(tr) >= period:
            atr_tr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_tr[i] = (atr_tr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(plus_dm) if np.all(atr_tr == 0) else 
                         np.convolve(plus_dm, np.ones(period)/period, mode='same') / atr_tr)
        minus_di = 100 * (np.zeros_like(minus_dm) if np.all(atr_tr == 0) else 
                          np.convolve(minus_dm, np.ones(period)/period, mode='same') / atr_tr)
        
        dx = np.zeros_like(high)
        dx[:] = np.nan
        denom = plus_di + minus_di
        dx = np.where(denom != 0, 100 * np.abs(plus_di - minus_di) / denom, 0)
        return dx
    
    period_adx = 14
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, period_adx)
    dx_1d = calculate_dx(high_1d, low_1d, close_1d, period_adx)
    adx_1d = np.zeros_like(dx_1d)
    adx_1d[:] = np.nan
    if len(dx_1d) >= period_adx:
        adx_1d[period_adx-1] = np.nanmean(dx_1d[period_adx-1:2*period_adx-1])
        for i in range(period_adx, len(dx_1d)):
            adx_1d[i] = (adx_1d[i-1] * (period_adx-1) + dx_1d[i]) / period_adx
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Elder Ray on 6h timeframe: EMA(13) of close
    ema13 = np.zeros(n)
    ema13[:] = np.nan
    if n >= 13:
        ema_multiplier = 2 / (13 + 1)
        ema13[0] = close[0]
        for i in range(1, n):
            ema13[i] = (close[i] - ema13[i-1]) * ema_multiplier + ema13[i-1]
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period average
    avg_volume = np.zeros(n)
    avg_volume[:] = np.nan
    if n >= 20:
        for i in range(20, n):
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Rising/falling power detection (3-bar change)
    bull_power_rising = np.zeros(n, dtype=bool)
    bear_power_falling = np.zeros(n, dtype=bool)
    bull_power_rising[:] = False
    bear_power_falling[:] = False
    for i in range(3, n):
        if not np.isnan(bull_power[i]) and not np.isnan(bull_power[i-3]):
            bull_power_rising[i] = bull_power[i] > bull_power[i-3]
        if not np.isnan(bear_power[i]) and not np.isnan(bear_power[i-3]):
            bear_power_falling[i] = bear_power[i] < bear_power[i-3]
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Trending market (ADX > 25): trade with Elder Ray momentum
            if adx_val > 25:
                # Long: Bull Power > 0 and rising + volume confirmation
                if bp > 0 and bull_power_rising[i] and volume_confirm:
                    position = 1
                    signals[i] = position_size
                # Short: Bear Power < 0 and falling + volume confirmation
                elif br < 0 and bear_power_falling[i] and volume_confirm:
                    position = -1
                    signals[i] = -position_size
            # Ranging market (ADX < 20): fade Elder Ray extremes
            elif adx_val < 20:
                # Calculate power volatility for dynamic thresholds
                lookback = min(50, i)
                if lookback >= 10:
                    bp_std = np.nanstd(bull_power[i-lookback:i]) if not np.all(np.isnan(bull_power[i-lookback:i])) else 1.0
                    br_std = np.nanstd(bear_power[i-lookback:i]) if not np.all(np.isnan(bear_power[i-lookback:i])) else 1.0
                    
                    # Long: Bear Power < -0.5*std and turning up (bullish divergence)
                    if br < -0.5 * br_std and not bear_power_falling[i] and volume_confirm:
                        position = 1
                        signals[i] = position_size
                    # Short: Bull Power > 0.5*std and turning down (bearish divergence)
                    elif bp > 0.5 * bp_std and not bull_power_rising[i] and volume_confirm:
                        position = -1
                        signals[i] = -position_size
            else:
                # Transition zone (20 <= ADX <= 25): no new positions
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power turns positive (momentum shift) or ADX weakens
            if br > 0 or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power turns negative (momentum shift) or ADX weakens
            if bp < 0 or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_ADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0