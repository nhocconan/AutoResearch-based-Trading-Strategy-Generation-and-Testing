#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Uses 1d ADX(14) to define regime: ADX>25 = trending (trade Elder Ray signals), ADX<20 = range (fade to EMA21)
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Entry: Long when Bull Power > 0 and rising (2-bar momentum) in trending regime OR when Bear Power < 0 and rising in range regime (mean reversion)
# Exit: Opposite signal or regime change
# Designed for low frequency (50-150 trades over 4 years) with clear bull/bear logic

name = "6h_ElderRay_1dADX_Regime_MeanRevTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d HTF data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX(14) calculation for regime detection
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
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # 2-bar momentum for confirmation
    bull_power_mom = bull_power - np.concatenate([[np.nan, np.nan], bull_power[:-2]])
    bear_power_mom = bear_power - np.concatenate([[np.nan, np.nan], bear_power[:-2]])
    
    # 6h EMA21 for mean reversion exits in range
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 21)  # Need ADX and EMA21
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(ema13[i]) or np.isnan(ema21[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_mom[i]) or np.isnan(bear_power_mom[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Trending regime: Elder Ray trend following
            if trending:
                # Long: Bull Power positive AND rising (momentum > 0)
                if bull_power[i] > 0 and bull_power_mom[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power negative AND rising (becoming less negative = momentum > 0)
                elif bear_power[i] < 0 and bear_power_mom[i] > 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging regime: Elder Ray mean reversion (fade extremes)
            elif ranging:
                # Long: Bear Power negative AND rising (extreme bearish exhaustion)
                if bear_power[i] < 0 and bear_power_mom[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power positive AND rising (extreme bullish exhaustion)
                elif bull_power[i] > 0 and bull_power_mom[i] > 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Transition regime (ADX 20-25) - stay flat
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if trending:
                # Exit trending long when Bull Power turns negative
                if bull_power[i] <= 0:
                    exit_long = True
            elif ranging:
                # Exit ranging long when price reaches EMA21 (mean reversion target)
                if close[i] >= ema21[i]:
                    exit_long = True
            else:
                # Transition regime - exit on any Elder Ray deterioration
                if bull_power[i] <= 0 or bear_power[i] >= 0:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if trending:
                # Exit trending short when Bear Power turns positive
                if bear_power[i] >= 0:
                    exit_short = True
            elif ranging:
                # Exit ranging short when price reaches EMA21 (mean reversion target)
                if close[i] <= ema21[i]:
                    exit_short = True
            else:
                # Transition regime - exit on any Elder Ray deterioration
                if bear_power[i] >= 0 or bull_power[i] <= 0:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals