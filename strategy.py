#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 12h ADX trend strength and volume confirmation.
# Choppiness Index (CHOP) identifies ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In trending regimes (CHOP < 38.2), we follow 12h ADX trend direction (ADX > 25).
# In ranging regimes (CHOP > 61.8), we mean-revert at Bollinger Band extremes (20, 2.0).
# Volume confirmation filters low-quality breakouts.
# Target: 20-50 trades per year (80-200 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour data for trend filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components for 12h
    def calculate_atr(high, low, close, period):
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros(len(high))
        if len(high) < period:
            return atr
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    def calculate_dmi(high, low, close, period):
        if len(high) < period:
            return np.zeros(len(high)), np.zeros(len(high)), np.zeros(len(high))
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Pad to same length
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        atr = calculate_atr(high, low, close, period)
        
        # Avoid division by zero
        plus_di = np.full(len(high), np.nan)
        minus_di = np.full(len(high), np.nan)
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (np.mean(plus_dm[i-period+1:i+1]) / atr[i]) * 100
                minus_di[i] = (np.mean(minus_dm[i-period+1:i+1]) / atr[i]) * 100
        
        dx = np.full(len(high), np.nan)
        for i in range(period, len(high)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
        
        adx = np.full(len(high), np.nan)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx, plus_di, minus_di
    
    adx_12h, plus_di_12h, minus_di_12h = calculate_dmi(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    plus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, plus_di_12h)
    minus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, minus_di_12h)
    
    # Choppiness Index on 4h (14-period)
    def calculate_chop(high, low, close, period):
        atr = calculate_atr(high, low, close, 1)
        if len(high) < period:
            return np.full(len(high), np.nan)
        
        # True range sum over period
        tr_sum = np.zeros(len(high))
        for i in range(period-1, len(high)):
            tr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.full(len(high), np.nan)
        lowest_low = np.full(len(high), np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.full(len(high), np.nan)
        for i in range(period-1, len(high)):
            if highest_high[i] != lowest_low[i] and tr_sum[i] > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Bollinger Bands for mean reversion in ranging markets (20, 2.0)
    def calculate_bb(close, period, std_dev):
        if len(close) < period:
            return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)
        
        sma = np.zeros(len(close))
        sma[:period-1] = np.nan
        sma[period-1] = np.mean(close[:period])
        for i in range(period, len(close)):
            sma[i] = sma[i-1] + (close[i] - close[i-period]) / period
        
        # Standard deviation
        bb_std = np.zeros(len(close))
        bb_std[:period-1] = np.nan
        for i in range(period-1, len(close)):
            bb_std[i] = np.std(close[i-period+1:i+1])
        
        upper = sma + (bb_std * std_dev)
        lower = sma - (bb_std * std_dev)
        return upper, sma, lower
    
    bb_upper, bb_middle, bb_lower = calculate_bb(close, 20, 2.0)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(chop[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(plus_di_12h_aligned[i]) or np.isnan(minus_di_12h_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        chop_val = chop[i]
        adx_val = adx_12h_aligned[i]
        plus_di_val = plus_di_12h_aligned[i]
        minus_di_val = minus_di_12h_aligned[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Trending regime: CHOP < 38.2
            if chop_val < 38.2:
                # Strong trend: ADX > 25
                if adx_val > 25:
                    # Long: +DI > -DI
                    if plus_di_val > minus_di_val and volume_confirm:
                        position = 1
                        signals[i] = position_size
                    # Short: -DI > +DI
                    elif minus_di_val > plus_di_val and volume_confirm:
                        position = -1
                        signals[i] = -position_size
            # Ranging regime: CHOP > 61.8
            elif chop_val > 61.8:
                # Mean reversion at Bollinger Bands
                if price <= bb_lower_val and volume_confirm:
                    position = 1
                    signals[i] = position_size
                elif price >= bb_upper_val and volume_confirm:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: regime change to ranging or mean reversion signal
            if chop_val > 61.8 and price >= bb_middle[i]:
                position = 0
                signals[i] = 0.0
            # Exit long in trend: ADX weakens or DI crossover
            elif chop_val < 38.2 and adx_val < 20:
                position = 0
                signals[i] = 0.0
            elif chop_val < 38.2 and minus_di_val > plus_di_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: regime change to ranging or mean reversion signal
            if chop_val > 61.8 and price <= bb_middle[i]:
                position = 0
                signals[i] = 0.0
            # Exit short in trend: ADX weakens or DI crossover
            elif chop_val < 38.2 and adx_val < 20:
                position = 0
                signals[i] = 0.0
            elif chop_val < 38.2 and plus_di_val > minus_di_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Choppiness_ADX_BB_Volume"
timeframe = "4h"
leverage = 1.0