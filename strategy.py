#!/usr/bin/env python3
"""
Hypothesis: 1d-based strategy using 1-week High-Low range breakout with volume confirmation and ADX trend filter.
In bull markets: buy breakout above weekly high with strong trend.
In bear markets: sell breakdown below weekly low with strong trend.
Uses weekly range to capture major structural breaks, reducing whipsaw vs daily levels.
Target: 15-25 trades/year to minimize fee drag while capturing significant moves.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly high and low (using previous week's values)
    weekly_high = np.full(len(high_1w), np.nan)
    weekly_low = np.full(len(low_1w), np.nan)
    
    for i in range(1, len(high_1w)):
        if not (np.isnan(high_1w[i-1]) or np.isnan(low_1w[i-1])):
            weekly_high[i] = high_1w[i-1]
            weekly_low[i] = low_1w[i-1]
    
    # Align weekly levels to daily timeframe
    weekly_high_daily = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_daily = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate ADX(14) for trend strength on daily data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period has no previous close
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        
        if len(tr) >= period:
            # Initial average
            atr[period-1] = np.mean(tr[:period])
            plus_dm_sum = np.sum(plus_dm[:period])
            minus_dm_sum = np.sum(minus_dm[:period])
            
            # Smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_sum = plus_dm_sum * (period-1) / period + plus_dm[i]
                minus_dm_sum = minus_dm_sum * (period-1) / period + minus_dm[i]
                
                plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
                minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
            
            # DX and ADX
            dx = np.zeros_like(tr)
            adx = np.zeros_like(tr)
            
            for i in range(period, len(tr)):
                di_sum = plus_di[i] + minus_di[i]
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
            
            if len(dx) >= 2*period-1:
                adx[2*period-2] = np.mean(dx[period-1:2*period-1])
                for i in range(2*period-1, len(tr)):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # need volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_daily[i]) or np.isnan(weekly_low_daily[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-day average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend strength: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long entry: break above weekly high with volume and strong trend
            if (close[i] > weekly_high_daily[i] and 
                vol_confirmed and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly low with volume and strong trend
            elif (close[i] < weekly_low_daily[i] and 
                  vol_confirmed and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below weekly low (reversal signal)
            if close[i] < weekly_low_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly high (reversal signal)
            if close[i] > weekly_high_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRangeBreakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0